"""Talk to Becky. One window, one prompt: type a request, or a command.

    list the files            -> she does it
    improve                   -> she trains a better version of her own brain (you approve the swap)
    review                    -> she inspects her own layers and speed
    compress                  -> she makes a smaller copy of her brain
    memory <words>            -> search her notes
    help / quit

Everything she does still goes through the guardian: her rules are signed by you, writes ask
you first, and every step is in the audit log. `improve` only ever *proposes* a better brain;
it replaces hers only if it scores better and you type y.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

HELP = """\
Type a request, for example:
    list the files
    read notes.txt
    write hello world to greeting.txt
    remember call mom
Or a command:
    improve    make her brain better (about 30 minutes; you decide whether to keep the result)
    review     she inspects her own layers and speed
    compress   make a smaller copy of her brain
    memory X   search her notes for X
    help       this list
    quit       leave
"""


class Chat:
    def __init__(
        self,
        run_request: Callable[[str], str],
        say: Callable[[str], None],
        ask: Callable[[str], str],
        improve: Callable[[], str],
        review: Callable[[], str],
        compress: Callable[[], str],
        search: Callable[[str], str],
    ):
        self.run_request, self.say, self.ask = run_request, say, ask
        self.commands = {"improve": improve, "review": review, "compress": compress}
        self.search = search

    def handle(self, line: str) -> bool:
        """Process one line. Returns False when the user wants to leave."""
        line = line.strip()
        if not line:
            return True
        word, _, rest = line.partition(" ")
        word = word.lower()
        if line.lower() in ("quit", "exit", "q"):
            return False
        if line.lower() in ("help", "?"):
            self.say(HELP)
        elif word == "memory" and rest.strip():
            self.say(self.search(rest.strip()))
        elif line.lower() in self.commands:
            self._guarded(self.commands[line.lower()])
        else:
            self._guarded(lambda: self.run_request(line))
        return True

    def _guarded(self, fn: Callable[[], str]) -> None:
        """Run something that may be slow or fail, without ever crashing the conversation."""
        try:
            self.say(fn())
        except KeyboardInterrupt:
            self.say("(stopped)")
        except SystemExit as exc:
            self.say(str(exc.code or exc))
        except Exception as exc:  # a failure is something to show, not a reason to exit
            self.say(f"That did not work: {type(exc).__name__}: {exc}")

    def loop(self) -> None:
        self.say("Becky is ready. Type help for ideas, quit to leave.\n")
        while True:
            try:
                line = self.ask("you> ")
            except (EOFError, KeyboardInterrupt):
                self.say("")
                return
            if not self.handle(line):
                self.say("Goodbye.")
                return
