"""Observe service admission while preserving SimPy generator delegation."""

from simpy.events import Timeout


def observe_service(process, on_start):
    """The compute driver yields its service timeout after resource admission."""
    try:
        event = next(process)
        while True:
            if isinstance(event, Timeout):
                on_start()
            try:
                value = yield event
            except GeneratorExit:
                process.close()
                raise
            except BaseException as error:
                event = process.throw(error)
            else:
                event = process.send(value)
    except StopIteration as completed:
        return completed.value
    finally:
        process.close()
