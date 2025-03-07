import time

import pipeline as pl
from clarifai.utils import video_utils

START_TIME = time.time()


class Counter(pl.Component):

  def __init__(self):
    super().__init__()
    self.count = 0

  def process(self, data):
    data.count = self.count
    data.create_time = pl.ts()
    self.count += 1


class Sleep(pl.Component):

  def __init__(self, seconds):
    super().__init__()
    self.seconds = seconds

  def process(self, data):
    time.sleep(self.seconds)
    data.sleeps = data.sleeps + 1 if hasattr(data, 'sleeps') else 1


class Print(pl.Component):

  def __init__(self, interval=None):
    super().__init__()
    self.interval = interval
    self.last_print = 0

  def process(self, data):
    if self.interval is not None and pl.ts() - self.last_print < self.interval:
      return
    print(
        f'{data.frame.pts}  {data.frame.time}  latency: {pl.ts() - data.create_time:.3f}  throughput: {meter.get():.3f}'
    )
    self.last_print = pl.ts()
    #print(f'{data.count}  latency: {pl.ts() - data.create_time:.3f}  throughput: {meter.get():.3f}')


class FrameReader(pl.Component):

  def __init__(self, video_frames_generator):
    super().__init__()
    self.video_frames_generator = video_frames_generator

  def process(self, data):
    data.frame = next(self.video_frames_generator)


#video = video_utils.stream_frames_from_file('demo.mp4')
def _teststream():
  while True:
    yield from video_utils.stream_frames_from_file('demo.mp4')


#def testgen():
#    for i in range(100000):
#        frame = types.SimpleNamespace()
#        frame.pts = i
#        frame.time = pl.ts()
#        yield frame
#video = testgen()


def test():
  video = _teststream()

  meter = pl.ThroughputMeter(print='meter')

  source = (FrameReader(video) >> pl.FixedRateLimiter(30))
  #source = FrameReader(video)

  last = (
      source >> Counter() >> pl.AdaptiveRateLimiter(meter, initial_rate=30, drop=True) >>
      pl.ThroughputMeter(print='first')
      #>> Sleep(0.01).num_threads(5)
      >> Sleep(0.05) >> meter >> Print())

  engine = pl.PipelineEngine()
  engine.add_component(last)
  engine.run()
