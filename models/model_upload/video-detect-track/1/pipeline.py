import itertools
import logging
import queue
import threading
import time
import types

_thread_local = threading.local()


class Drop(Exception):
  '''
  Exception to signal that the current item was dropped/skipped by the component.
  '''


class State(enum.Enum):
  NONE = 0
  QUEUED = 1
  STARTED = 2
  DROPPED = 3
  COMPLETED = 4
  FAILED = 5


class Item:

  def __init__(self):
    self.states = {}
    self.error = None
    self.data = types.SimpleNamespace()


class Pipeline:

  def __init__(self, components=None):
    self.work_buffer = []  # buffer for work items
    self.components = []  # list of all components
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

  def callback(self, item, component_id):
    # called upon item completion, failure, or drop by a component to signal a state change
    self.changed_event.set()

  def run(self):
    for component in self.components:
      component.start()
    while True:
      self.changed_event.wait()
      self.changed_event.clear()
      self._queue_components()
      self._cleanup()

  def _queue_components(self):
    for component in self.components:
      component_id = component.id
      # go through items oldest to most recent to check if the component can be scheduled
      for item in self.work_buffer:
        if item.states.get(component_id, 0) >= State.QUEUED:
          # already scheduled this item
          continue
        if all(item.states[dep] == State.COMPLETED for dep in component.dependencies):
          component.enqueue(state)
        else:
          break  # go to next component, this one is blocked for the next state

  def _cleanup(self):
    # cleanup all items that have no work left
    while self.work_buffer:
      item = self.work_buffer[0]
      if any(s in (State.QUEUED, State.STARTED) for s in item.states.values()):
        # this item, or others after it that are blocked by ordering, can still be processed
        break
      if item.error:
        logging.error("State failed with error: %s", item.error)
      # remove item from buffer
      logging.debug("Cleaning up item %s", item)
      self.work_buffer.pop(0)


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

  def enqueue(self, item):
    item.states[self.id] = State.QUEUED
    self.queue.put(item)

  def run(self):
    while True:
      try:
        item = self.queue.get()  # blocking get for the next item
        item.states[self.id] = State.STARTED
        try:
          self.process(item.data)
        except Drop:
          item.states[self.id] = State.DROPPED
        except Exception as e:
          item.error = e
          item.states[self.id] = State.FAILED
        else:
          item.states[self.id] = State.COMPLETED
        finally:
          self.pipeline.callback(item, self.id)
          self.queue.task_done()
      except Exception:
        logging.exception('Internal error in component %s', self.id)
        time.sleep(1)

  def process(self, item_data):
    raise NotImplementedError("Subclasses must implement this method.")
