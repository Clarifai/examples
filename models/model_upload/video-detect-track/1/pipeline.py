import enum
import itertools
import logging
import os
import queue
import threading
import time
import types

_thread_local = threading.local()

logging.basicConfig(level=os.environ.get('LOGLEVEL', 'INFO').upper())


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

  def __gt__(self, other):
    return self.value > other.value

  def __ge__(self, other):
    return self.value >= other.value


_START_TIME = time.monotonic()


def monotonic():
  return time.monotonic() - _START_TIME


class Item:

  _ID_COUNTER = itertools.count()

  def __init__(self):
    self.num = next(Item._ID_COUNTER)
    self.states = {}
    self.error = None
    self.data = types.SimpleNamespace()

  def set_state(self, component_id, state):
    logging.debug("%f %s %d, -> %s", monotonic(), component_id, self.num, state)
    self.states[component_id] = state


class Pipeline:

  def __init__(self, components=[]):
    self.work_buffer = []  # buffer for work items
    self.components = []  # list of all components
    self.changed_event = threading.Event()
    self.max_buffer_size = 10  # TODO how to add elements?
    for component in components:
      self.add_component(component)

  def add_component(self, component):
    if component not in self.components:
      self.components.append(component)
      component.pipeline = self

  def put(self, item):
    self.work_buffer.append(item)
    self.changed_event.set()

  def __enter__(self):
    if getattr(_thread_local, 'pipeline', None) is not None:
      raise Exception("Nested pipelines are not supported.")
    _thread_local.pipeline = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert _thread_local.pipeline is self
    _thread_local.pipeline = None

  def run(self):
    for component in self.components:
      logging.debug("Starting component %s", component.id)
      component.start()
    self.changed_event.set()
    while True:
      self.changed_event.wait()
      self.changed_event.clear()
      while len(self.work_buffer) < self.max_buffer_size:
        self.put(Item())  # add new items; TODO how to add elements?
      self._queue_components()
      self._cleanup()

  def callback(self, item, component_id):
    # called upon item completion, failure, or drop by a component to signal a state change
    self.changed_event.set()

  def _queue_components(self):
    for component in self.components:
      component_id = component.id
      # go through items oldest to most recent to check if the component can be scheduled
      for item in self.work_buffer:
        if item.states.get(component_id, State.NONE) >= State.QUEUED:
          # already scheduled this item, check next in buffer list
          continue
        if all(item.states.get(dep) == State.COMPLETED for dep in component.dependencies):
          component.enqueue(item)
        else:
          break  # go to next component, this one is blocked

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

  _ID_COUNTER = map(str, itertools.count())

  def __init__(self, num_threads=1, queue_size=1):
    self.id = self.__class__.__name__ + '-' + next(Component._ID_COUNTER)
    self.queue = queue.Queue(maxsize=queue_size)
    self.dependencies = set()
    self.num_threads = num_threads
    self.threads = []
    if getattr(_thread_local, 'pipeline', None) is not None:
      _thread_local.pipeline.add_component(self)

  def start(self):
    for i in range(self.num_threads):
      thread = threading.Thread(target=self.run, daemon=True, name=self.id + '-' + str(i))
      self.threads.append(thread)
      thread.start()

  def depends_on(self, other):
    self.dependencies.add(other.id)
    if getattr(_thread_local, 'pipeline', None) is not None:
      _thread_local.pipeline.add_component(self)

  def __rshift__(self, other):
    other.depends_on(self)
    return other

  def enqueue(self, item):
    try:
      self.queue.put(item, block=False)
    except queue.Full:
      pass
    else:
      item.set_state(self.id, State.QUEUED)

  def run(self):
    while True:
      try:
        item = self.queue.get()  # blocking get for the next item
        item.set_state(self.id, State.STARTED)
        try:
          self.process(item.data)
        except Drop:
          item.set_state(self.id, State.DROPPED)
        except Exception as e:
          item.error = e
          item.set_state(self.id, State.FAILED)
        else:
          item.set_state(self.id, State.COMPLETED)
        finally:
          self.pipeline.callback(item, self.id)
          self.queue.task_done()
      except Exception:
        logging.exception('Internal error in component %s', self.id)
        time.sleep(1)

  def process(self, item_data):
    raise NotImplementedError("Subclasses must implement this method.")
