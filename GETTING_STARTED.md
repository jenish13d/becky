# Getting started: run your own AI, step by step (Windows)

## The easy way (two double-clicks)

1. Install Python from python.org and **tick "Add python.exe to PATH"** (skip if you already have it).
2. Unzip the project anywhere (or `git clone https://github.com/jenish13d/becky.git`).
3. Double-click **`setup.bat`**. It installs everything (a few minutes, once).
4. Double-click **`start.bat`**. The first time, it asks you to choose your passphrase (nothing
   shows while you type; that is normal). Then you are talking to Becky:

```
you> list the files
you> read notes.txt
you> write hello world to greeting.txt        <- she asks first; type y to allow
you> improve                                  <- she trains a better version of herself
you> review                                   <- she inspects her own layers and speed
you> compress                                 <- a smaller copy of her brain
you> memory dentist                           <- search her notes
you> quit
```

`improve` takes about 30 minutes. When it finishes it shows `current brain X% -> candidate Y%`
and asks whether to keep it. She never swaps her own brain without your yes, and your old
brain is kept as `ouroboros\brain.previous.pt`.

Everything below is the long version with every individual command, in case you want it.

---

Everything here runs **on your own PC**. There is no Ollama, no Llama, no cloud model and no
API key. Becky's brain is a small model trained from scratch by this project; its weights are
the file `ouroboros\brain.pt` inside the project.

Copy each command exactly. Lines starting with `REM` are explanations, not commands.

---

## Part 1: Install (about 10 minutes, one time)

**Step 1. Open Command Prompt.** Press the Windows key, type `cmd`, press Enter.

**Step 2. Check Python.**
```
python --version
```
You need 3.11 or newer. If you get an error, install Python from python.org and tick
**"Add python.exe to PATH"** on the first screen, then open a new Command Prompt.

**Step 3. Check Git.**
```
git --version
```
If you get an error, install Git from git-scm.com (accept the defaults), then open a new Command Prompt.

**Step 4. Download the project.**
```
cd %USERPROFILE%
git clone https://github.com/jenish13d/becky.git
cd becky
```
If you already downloaded it before, update it instead:
```
cd %USERPROFILE%\becky
git pull
```

**Step 5. Install what it needs.**
```
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt -e .
```

**Step 6. Check everything works.**
```
python -m pytest
```
This takes a few minutes. The last line should say `passed` with no `failed`.
If it does, the whole project works on your machine.

> From now on, always run commands from inside the `becky` folder
> (`cd %USERPROFILE%\becky`).

---

## Part 2: Make Becky answer only to you (one time)

**Step 7. Choose your owner passphrase.**
```
python -m ouroboros init-owner
```
It asks for a passphrase twice. **Nothing appears while you type; that is normal.**

* Use four or more random words (like `river-moss-copper-velvet`) or 16+ characters.
* Do **not** use a keyboard pattern like `asdfghjkl13M`; it is rejected, and anything you
  have typed into a chat should be treated as public.
* Write it down somewhere only you can reach. It is stored nowhere. If you lose it, nobody
  can change Becky's rules again and you would run Step 7 again from scratch.

What it does: it creates a signing key (locked by your passphrase) in `C:\Users\<you>\.becky\`.
Becky refuses to start unless her rules are signed by that key, so only you can change them.

---

## Part 3: Talk to Becky

**Step 8. Create her workspace and two sample files.** Becky can only see this one folder.
```
mkdir becky_workspace
echo meeting at noon> becky_workspace\notes.txt
echo buy milk and bread> becky_workspace\todo.txt
```

**Step 9. Ask her things.**
```
python -m ouroboros becky "list the files"
python -m ouroboros becky "read notes.txt"
python -m ouroboros becky "write hello world to greeting.txt"
python -m ouroboros becky "remember call mom"
```
* Reading and listing happen immediately.
* **Writing asks you first.** She prints exactly what she wants to do and waits for `y` or `n`.
  Type `y` to allow it. Anything else refuses it.
* Check the result: `type becky_workspace\greeting.txt`

**What she understands.** Becky is a small model with one narrow skill: turning a plain
request into one of four actions. These phrasings work reliably:

| To do this | Say one of these |
| --- | --- |
| List files | `list the files` · `what files are here` · `show me the files` · `ls` |
| Read a file | `read notes.txt` · `show me notes.txt` · `open notes.txt` · `what does notes.txt say` · `cat notes.txt` |
| Write a file | `write hello world to greeting.txt` · `save hello world in greeting.txt` · `create greeting.txt containing hello world` |
| Save a note | `remember call mom` · `note call mom` · `make a note: call mom` |

**Known limit: keep contents short.** She copies words back reliably up to about **4 words** (5 mostly work, 6 is shaky, 8 or more fails). So `read notes.txt` on a long file will return a garbled version, and `write` is best used with short text. This is what to improve next: her training only ever contained short contents.

File names should be letters, digits, `_` or `-` ending in `.txt` or `.md`. Anything else
(`hello`, `delete x`, `who are you`) gets the answer *"I can only list files, read a file,
write a file, or remember a note."* She cannot chat, reason, or follow new kinds of
instructions; she is not a general assistant.

**Step 10. See everything she did.** Every decision is written to a tamper-evident log.
```
type %USERPROFILE%\.becky\audit.jsonl
python -m ouroboros audit
```
The second command says `audit log intact` or `TAMPERING DETECTED`.

**Step 11. The kill switch.** To stop her immediately, create this file:
```
type nul > %USERPROFILE%\.becky-stop
```
She refuses to act while it exists. Delete it to resume: `del %USERPROFILE%\.becky-stop`

**Step 12. Change how much she asks you (needs your passphrase).**
```
python -m ouroboros amend --set write_file=allow
python -m ouroboros amend --set write_file=ask
python -m ouroboros amend --reset
```
`allow` means no question asked, `ask` means she asks you, `deny` means never.
This changes how much she asks. It can never give her a shell, internet access, or a way out of
her workspace; those limits are in the code.

---

## Part 4: Let her run on a schedule (optional)

**Step 13. Give her a task list.** Only you can write this; she cannot add to it.
```
python -m ouroboros task-add daily "list the files"
python -m ouroboros task-add notes "read notes.txt"
```

**Step 14. Run the list once, by hand.**
```
python -m ouroboros hub
```
A report is saved in `%USERPROFILE%\.becky\reports\`. There is nobody to approve anything during
a scheduled run, so anything that would normally ask you (like writing) is automatically
refused, unless you did Step 12's `write_file=allow`.

**Step 15. Run it every day.** This prints the exact command for your PC; you paste it yourself.
```
python -m ouroboros hub --print-schedule
```
To remove it later: `schtasks /delete /tn "Becky"`

---

## Part 5: Let her improve herself

**Step 16. Improve Becky's brain.** This is the "keep improving, and when stuck, change approach" loop.
```
python -m ouroboros brain-improve --rounds 6 --steps 400 --eval-n 100 --time-budget 3600
```
It takes about an hour on a normal CPU. Each round it trains a candidate and scores it on fresh
test examples it has never seen. If the candidate is not clearly better, the round counts as
**stuck** and the next round switches to a different tactic:

1. **fresh**: more training on newly generated data
2. **failures**: find the champion's own mistakes and train on those harder
3. **diversify**: the same requests reworded many ways (`can you ...`, `... for me`)
4. **polish**: a much lower learning rate and twice the steps
5. **grow**: add a layer, keeping everything learned, and train it

Any real improvement goes back to tactic 1. It stops only when all five fail in a row, and then
tells you it has plateaued and what would help (more varied data, or a bigger budget).

**This does not change Becky's brain by itself.** The improved brain is saved to
`runs\brain-improve\champion.pt`. Step 17 is where you decide.

**Step 17. Adopt the improved brain (your decision).**
```
python -m ouroboros brain-improve --rounds 6 --steps 400 --eval-n 100 --time-budget 3600 --apply
```
After the run it shows `current brain X% -> candidate Y%` and asks `[y/N]`. Type `y` to make it
Becky's brain; the old one is kept as `ouroboros\brain.previous.pt`. It refuses outright if the
candidate is not better. Check the score for yourself any time with
`python -m ouroboros brain-eval`.

---

## Part 6: Make her smaller, faster, and tamper-proof

**Step 18. Shrink her brain** (about 2.6x smaller, same accuracy):
```
python -m ouroboros compress --out brain-int8.pt
```
It prints the size and accuracy before and after, and says `Safe to use` or `NOT safe`. To run her
from the small file: `python -m ouroboros becky "list the files" --brain-path brain-int8.pt`.
This saves disk space only; it does not use less memory.

**Step 19. Let her review herself** (changes nothing):
```
python -m ouroboros review
```
It shows where her parameters are, how fast she runs, and what each layer is worth. Any
suggestion is yours to act on.

**Step 20. Search her memory and remove repeats:**
```
python -m ouroboros memory-search "dentist"
python -m ouroboros memory-dedupe
```
This is fuzzy text matching, not understanding: `buy milk` finds `Buy milk!` but not `purchase dairy`.

**Step 21. Lock her down with an integrity anchor** (needs your passphrase):
```
python -m ouroboros anchor
python -m ouroboros verify-anchor
```
`anchor` records fingerprints of her brain, rules and code and signs them. From then on she
refuses to start if anything changed, and names the file. **After you update on purpose**
(`git pull`, or adopting an improved brain in Step 17) she will refuse until you look at what
changed and run `python -m ouroboros anchor` again. That is how it is meant to work.

---

## Part 7: The other pieces

**What "rewrite" and "apply" mean in `selfedit`.**
```
python -m ouroboros selfedit
python -m ouroboros selfedit --apply
```
* **Without `--apply`** the program works on a *copy* of one small file
  (`ouroboros\organism.py`), saves the improved copy under `runs\selfedit\`, and the original
  is never touched. Nothing is rewritten.
* **With `--apply`** it does the same, then shows you the exact lines it wants to change and asks
  `[y/N]`. Only if you type `y` does it overwrite the small marked section of that one file.
* The "self" here is a **toy formula** that the program learns to fit. It is a demonstration of
  the safe approval mechanism, not Becky and not her brain.

**Spiking neural network** (brain-style neurons that fire spikes):
```
python -m ouroboros snn-demo
```

**The from-scratch language model** (separate from Becky's brain; learns Shakespeare):
```
python -m ouroboros train --steps 2000
python -m ouroboros evolve --generations 8 --children 3 --steps 300 --seed-steps 600 --max-params 800000
python -m ouroboros sample --ckpt runs\evo\champion.pt --prompt "ROMEO:" --tokens 400
```
`evolve` also changes approach when stuck: it widens its search before giving up.

---

## If something goes wrong

| What you see | What to do |
| --- | --- |
| `'python' is not recognized` | Reinstall Python and tick "Add python.exe to PATH", or use `py` instead of `python`. |
| `'git' is not recognized` | Install Git from git-scm.com, then open a new Command Prompt. |
| `No owner passphrase is set up yet` | Run Step 7. |
| `integrity check FAILED` | Something she is built from changed. If you meant that, run Step 21's `anchor` again. If not, restore the file (`git checkout -- <file>`). |
| `Refusing to start: ... not signed by the owner` | Her rules file was edited by hand. Run `python -m ouroboros amend --reset` with your passphrase. |
| `kill switch engaged` | Delete `%USERPROFILE%\.becky-stop` (Step 11). |
| `no trained brain` | Run `git pull`, or train one: `python -m ouroboros brain-train`. |
| She answers "That failed." | The file name is wrong or not in `becky_workspace`. Run `list the files`. |
| She answers "I can only ..." | Rephrase using the table in Step 9. |
| Any other error | Copy the full message and send it to me. |

## What this is, honestly

A small, real, from-scratch AI that you own and that runs entirely on your machine, under rules
that only you can change. It is **not** general intelligence, it does not understand open-ended
requests, and it cannot become more than the narrow job it was trained for. What it does have is
a working loop that improves it, notices when it is stuck, and changes tactic, with you holding
the approval switch.
