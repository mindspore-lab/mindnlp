import mindspore.multiprocessing as mp
ctx = mp.get_context("spawn")

def lanuch_wrapper(*wargs, **kwargs):
    mp.set_start_method('spawn', force=True)
    process = mp.Process(target=launcher, args=args, kwargs=kwargs)
    process.start()
    process.join()