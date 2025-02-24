import itertools
import logging
import queue
import threading
import time

_thread_local = threading.local()


class Drop(Exception):
  '''
  Exception to signal that the current state was dropped/skipped by the component.
  '''


class State:

  def __init__(self):
    self.queued = set()
    self.started = set()
    self.completed = set()
    self.dropped = set()
    self.failed = set()
    self.data = {}
    self.error = None


class Pipeline:

  def __init__(self, components=None):
    self.state_buffer = []
    self.components = []
    self.changed_event = threading.Event()
    for component in components:
      self.add(component)

  def add(self, component):
    if component not in self.components:
      self.components.append(component)
      component.pipeline = self

  def __enter__(self):
    if getattr(_thread_local, 'pipeline', None) is not None:
      raise Exception("Nested pipelines are not supported.")
    _thread_local.pipeline = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert _thread_local.pipeline is self
    _thread_local.pipeline = None

  def completed_callback(self, state, component_id):
    # called upon state completion, failure, or drop by a component
    self.changed_event.set()

  def run(self):
    for component in self.components:
      component.start()
    while True:
      self.changed_event.wait()
      self.changed_event.clear()
      for component in self.components:
        # go through states oldest to most recent
        for state in self.state_buffer:
          if component in state.queued:  # already scheduled or completed
            continue
          if all(dep in state.completed for dep in component.dependencies):
            component.enqueue(state)
          else:
            break  # go to next component, this one is blocked for the next state


class Component:

  _ID_COUNTER = itertools.count()

  def __init__(self, num_threads=1):
    self.id = self.__class__.__name__ + '-' + next(Component._ID_COUNTER)
    self.queue = queue.Queue()
    self.dependencies = set()
    self.num_threads = num_threads
    self.threads = []
    if getattr(_thread_local, 'pipeline', None) is not None:
      _thread_local.pipeline.add(self)

  def start(self):
    for i in range(self.num_threads):
      thread = threading.Thread(target=self.run, daemon=True, name=self.id + '-' + str(i))
      self.threads.append(thread)
      thread.start()

  def depends_on(self, other):
    self.dependencies.add(other.id)
    if getattr(_thread_local, 'pipeline', None) is not None:
      _thread_local.pipeline.add(self)

  def __rshift__(self, other):
    other.depends_on(self)
    return other

  def enqueue(self, state):
    state.queued.add(self.id)
    self.queue.put(state)

  def run(self):
    while True:
      try:
        state = self.queue.get()  # blocking get for the next state
        state.started.add(self.id)
        try:
          self.process(state)
        except Drop:
          state.dropped.add(self.id)
        except Exception as e:
          state.failed.add(self.id)
          state.error = e
        else:
          state.completed.add(self.id)
        finally:
          self.queue.task_done()
          self.pipeline.completed_callback(state, self.id)
      except Exception:
        logging.exception('Internal error in component %s', self.id)
        time.sleep(1)

  def process(self, state):
    raise NotImplementedError("Subclasses must implement this method.")
