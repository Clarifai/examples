import enum
import itertools
import logging
import os
import queue
import threading
import time
import traceback
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


def ts():
  return time.monotonic() - _START_TIME


class Item:

  _ID_COUNTER = itertools.count()

  def __init__(self):
    self.id = next(Item._ID_COUNTER)
    self.states = {}
    self.error = None
    self.data = types.SimpleNamespace()

  def __repr__(self):
    return f'Item({self.id})'

  def set_state(self, component_id, state):
    logging.debug("%f %s %s, -> %s", ts(), component_id, self, state)
    self.states[component_id] = state


class PipelineEngine:

  def __init__(self, max_buffer_size=100):
    self.work_buffer = []  # buffer for work items
    self.components = []  # list of all components
    self.changed_event = threading.Event()
    self.max_buffer_size = max_buffer_size
    self.running = True

  def add_component(self, component):
    if component not in self.components:
      self.components.append(component)
      component.engine = self

  def __enter__(self):
    if getattr(_thread_local, 'engine', None) is not None:
      raise Exception("Nested pipeline engines are not supported.")
    _thread_local.engine = self
    return self

  def __exit__(self, exc_type, exc_val, exc_tb):
    assert _thread_local.engine is self
    _thread_local.engine = None

  def run(self):
    self._verify_components()
    try:
      for component in self.components:
        logging.debug("Starting component %s", component.id)
        component.start()
      self.changed_event.set()
      while True:
        self.changed_event.wait()
        self.changed_event.clear()
        self._start_items()
        self._schedule_components()
        self._cleanup()
        self._start_items()
    finally:
      self.running = False

  def callback(self, item, component_id):
    # called upon item completion, failure, or drop by a component to signal a state change
    self.changed_event.set()

  def _start_items(self, new=1):
    while (len(self.work_buffer) < self.max_buffer_size and
           (len(self.work_buffer) < new or
            any(s.value for s in self.work_buffer[-new].states.values()))):
      self.work_buffer.append(Item())
      self.changed_event.set()

  def _schedule_components(self):
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
    num_remove = 0
    for item in self.work_buffer:
      if any(s in (State.QUEUED, State.STARTED) for s in item.states.values()):
        # this item, or others after it that are blocked by ordering, can still be processed
        break
      if item.error:
        logging.error(
            f"Error in component:\n{''.join(traceback.format_exception(None, item.error, item.error.__traceback__))}"
        )
      num_remove += 1
    if num_remove:
      logging.debug("cleaning %s", self.work_buffer[:num_remove])
      del self.work_buffer[:num_remove]

  def _verify_components(self):
    # check dependency ids for unknown components outside of the engine context
    all_ids = {c.id for c in self.components}
    for component in self.components:
      for dep_id in component.dependencies:
        if dep_id not in all_ids:
          raise Exception(
              f"Component {component.id} depends on unknown component {dep_id} not part of the engine."
          )
    # check for cycles
    self._check_cycles()

  def _check_cycles(self):
    # https://en.wikipedia.org/wiki/Topological_sorting#Kahn's_algorithm
    in_degree = {c.id: 0 for c in self.components}
    for c in self.components:
      for dep_id in c.dependencies:
        in_degree[dep_id] += 1
    queue = [cid for cid, d in in_degree.items() if d == 0]
    while queue:
      cid = queue.pop()
      c = next(c for c in self.components if c.id == cid)
      for dep_id in c.dependencies:
        in_degree[dep_id] -= 1
        if in_degree[dep_id] == 0:
          queue.append(dep_id)
    if any(d != 0 for d in in_degree.values()):
      raise Exception("Dependency cycle detected.")

  def components_between(self, a, b=None):
    '''
    Return list of components between a and b (exclusive) according to the dependency graph.
    If b is None, return all components downstream of a.
    '''
    nodes = {a}
    visited = set()

    def dfs(node, path):
      if isinstance(node, str):
        node = next(c for c in self.components if c.id == node)
      if node in visited:
        return
      visited.add(node)
      if node in nodes:  # found a path that leads to a node that leads to a
        nodes.update(path)
        return
      path.append(node)
      for c in node.dependencies:
        dfs(c, path)
      path.pop()

    if b is not None:
      dfs(b, [])
    else:
      for c in self.components:
        dfs(c, [])

    nodes.discard(a)
    nodes.discard(b)
    return [c for c in self.components if c in nodes]


class Component:

  _ID_COUNTER = map(str, itertools.count())

  def __init__(self, num_threads=1, queue_size=1):
    self.id = self.__class__.__name__ + '-' + next(Component._ID_COUNTER)
    self.queue = queue.Queue(maxsize=queue_size)
    self.dependencies = set()
    self.num_threads = num_threads
    self.average_qsize = 0
    self.threads = []
    if getattr(_thread_local, 'engine', None) is not None:
      _thread_local.engine.add_component(self)

  def start(self):
    for i in range(self.num_threads):
      thread = threading.Thread(target=self.run_loop, daemon=True, name=self.id + '-' + str(i))
      self.threads.append(thread)
      thread.start()

  def depends_on(self, other):
    self.dependencies.add(other.id)
    if getattr(_thread_local, 'engine', None) is not None:
      _thread_local.engine.add_component(self)

  def __rshift__(self, other):
    other.depends_on(self)
    return other

  def enqueue(self, item):
    try:
      self.average_qsize = self.average_qsize * 0.9 + self.queue.qsize() * 0.1
      self.queue.put(item, block=False)
    except queue.Full:
      pass
    else:
      item.set_state(self.id, State.QUEUED)

  def run_loop(self):
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
          self.engine.callback(item, self.id)
          self.queue.task_done()
      except Exception:
        logging.exception('Internal error in component %s', self.id)
        time.sleep(1)

  def process(self, item_data):
    raise NotImplementedError("Subclasses must implement this method.")


class ThroughputMeter(Component):

  def __init__(self, alpha=0.01):
    super().__init__()
    self.lock = threading.Lock()
    self.alpha = alpha
    self.alpha_inv = 1 / alpha
    self.reset()

  def process(self, data):
    self.update(1)

  def reset(self):
    self.start_time = ts()
    self.throughput = 0
    self.num_updates = 0

  def get(self):
    return self.throughput

  def update(self, count):
    if not count: return
    now = ts()
    with self.lock:
      current = count / (now - self.start_time)
      # a increases to alpha according to schedule for unbiased sample (i.e. not influenced by 0 init throughput value)
      if self.num_updates > self.alpha_inv:
        a = self.alpha
      else:
        a = 1 / min(self.alpha_inv, self.num_updates + 1)  # 1 -> alpha
      self.throughput = self.throughput * (1 - a) + current * a
      self.num_updates += 1
      self.start_time = now


class FixedRateLimiter(Component):

  def __init__(self, rate=15):
    super().__init__()
    self.engine = getattr(_thread_local, 'engine', None)
    self.rate = rate
    self.last_time = 0

  def process(self, data):
    elapsed = ts() - self.last_time
    interval = 1.0 / self.rate
    if elapsed < interval:
      time.sleep(interval - elapsed)
    self.last_time = ts()


class AdaptiveRateLimiter(Component):

  def __init__(self, meter, initial_rate=30, delta=0.1, target_qsize=0.1, drop=False):
    super().__init__()
    self.meter = meter
    self.rate = initial_rate
    self.delta = delta
    self.target_qsize = target_qsize
    self.drop = drop
    self.last_time = 0
    self._downstream_components = None

  def process(self, data):
    if self._downstream_components is None:
      self._downstream_components = self.engine.components_between(self, self.meter)
      assert self not in self._downstream_components

    elapsed = ts() - self.last_time

    throughput = self.meter.get()
    qsize = max(c.average_qsize for c in self._downstream_components)

    logging.debug("RATE: %s  THROUGHPUT: %s  QSIZE: %s", self.rate, throughput, qsize)

    # adjust rate based on throughput and queue size
    if throughput != 0:
      if qsize < self.target_qsize:
        self.rate = throughput * (1 + self.delta)
      else:
        self.rate = throughput * (1 - self.delta)

    # sleep for the remaining time according to the rate
    interval = 1.0 / self.rate
    if elapsed < interval:
      if self.drop:
        raise Drop("Dropping item")
      time.sleep(interval - elapsed)
    self.last_time = ts()
