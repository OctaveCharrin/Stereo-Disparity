class TupleTransform:
  """
  Transform the whole 4-tuple with the specified transform
  """
  def __init__(self, transform) -> None:
    self.transform = transform

  def __call__(self, imgs, dsp_maps):
    imgl, imgr = imgs
    print(type(imgl), type(imgr))
    print(type(dsp_maps[0]))

    imgs = (self.transform(imgl), self.transform(imgr))
    return (imgs, self.transform(dsp_maps[0]))