import inspect
from typing import Any, Callable, Optional, Sequence, Union

from napari.qt.threading import FunctionWorker, GeneratorWorker, create_worker


def do_work_async(
    function: Callable,
    return_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
    yield_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
    error_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
    async_disabled: bool = False,
    **kwargs,
) -> FunctionWorker:
    if async_disabled:
        return _do_work_sync(
            function=function,
            return_callback=return_callback,
            yield_callback=yield_callback,
            error_callback=error_callback,
            **kwargs,
        )
    worker = create_worker(function, **kwargs)
    _connect_callback(worker.returned, return_callback)
    if isinstance(worker, GeneratorWorker):
        _connect_callback(worker.yielded, yield_callback)
    _connect_callback(worker.errored, error_callback)
    worker.start()
    return worker


def _connect_callback(event, callback: Optional[Union[Callable, Sequence[Callable]]]):
    if callback is None:
        return
    elif isinstance(callback, Callable):
        event.connect(callback)
    else:
        for c in callback:
            event.connect(c)


def _do_work_sync(
    function: Callable,
    return_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
    yield_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
    error_callback: Optional[Union[Callable, Sequence[Callable]]] = None,
    **kwargs,
):
    try:
        if inspect.isgeneratorfunction(function):
            gen = function(**kwargs)
            try:
                while True:
                    item = next(gen)
                    _call_callback(yield_callback, item)
            except StopIteration as e:
                result = e.value
        else:
            result = function(**kwargs)

        _call_callback(return_callback, result)
    except Exception:
        _call_callback(error_callback)
        raise


def _call_callback(
    callback: Optional[Union[Callable, Sequence[Callable]]], args: Any = None
):
    if callback is None:
        return
    elif isinstance(callback, Callable):
        callback = [callback]

    for c in callback:
        if args is None:
            c()
        # elif isinstance(args, (list, tuple)):
        #     c(*args)
        else:
            c(args)
