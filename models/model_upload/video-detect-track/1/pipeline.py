import itertools
import logging
import queue
import threading
import time
import types

_thread_local = threading.local()


class Drop(Exception):
  '''
  Exception to signal that the current state was dropped/skipped by the component.
  '''


class States(enum.Enum):
  NONE = 0
  QUEUED = 1
  STARTED = 2
  DROPPED = 3
  COMPLETED = 4
  FAILED = 5


class State:

  def __init__(self):
    self.component_states = {}
    self.error = None
    self.data = types.SimpleNamespace()


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

  def callback(self, state, component_id):
    # called upon state completion, failure, or drop by a component to signal a state change
    self.changed_event.set()

  def run(self):
    for component in self.components:
      component.start()
    while True:
      self.changed_event.wait()
      self.changed_event.clear()
      self._queue_components()
      self._cleanup_states()

  def _queue_components(self):
    for component in self.components:
      component_id = component.id
      # go through states oldest to most recent to check if the component can be scheduled
      for state in self.state_buffer:
        if state.component_states.get(component_id, 0) >= States.QUEUED:
          # already scheduled this state
          continue
        if all(dep in state.completed for dep in component.dependencies):
          component.enqueue(state)
        else:
          break  # go to next component, this one is blocked for the next state

  def _cleanup_states(self):
    # cleanup all states that have no work left
    while self.state_buffer:
      state = self.state_buffer[0]
      if any(s in (States.QUEUED, States.STARTED) for s in state.component_states.values()):
        # this state, or others after it that are blocked by ordering, can still be processed
        break
      if state.error:
        logging.error("State failed with error: %s", state.error)
      # remove state from buffer
      logging.debug("Cleaning up state %s", state)
      self.state_buffer.pop(0)


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
    state.component_states[self.id] = States.QUEUED
    self.queue.put(state)

  def run(self):
    while True:
      try:
        state = self.queue.get()  # blocking get for the next state
        state.states[self.id] = States.STARTED
        try:
          self.process(state.data)
        except Drop:
          state.states[self.id] = States.DROPPED
        except Exception as e:
          state.error = e
          state.states[self.id] = States.FAILED
        else:
          state.states[self.id] = States.COMPLETED
        finally:
          self.pipeline.callback(state, self.id)
          self.queue.task_done()
      except Exception:
        logging.exception('Internal error in component %s', self.id)
        time.sleep(1)

  def process(self, state_data):
    raise NotImplementedError("Subclasses must implement this method.")
