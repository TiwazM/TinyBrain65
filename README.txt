TinyBrain65

A tiny language model running directly on the **MEGA65**, answering short questions about its hardware and BASIC. A retro-computing experiment with just **375,600 parameters**.

![TinyBrain65 answering a sound question](XEMU-HARDWARE.png)

- 4-bit quantized weights, with an optimized C and 45GS02 assembly runtime.
- Inference runs entirely on the MEGA65, using Attic RAM. No network or PC required after training.
- Streaming answers in a green terminal with scrolling history and a little glow.
- Three transformer layers, width 96, four attention heads and a 1,024-token vocabulary.

**V6.1 is an experimental update.** It improves several hardware answers, but some older answers regress. Unfamiliar wording can still produce confident nonsense. Each question starts fresh; the visible history is not model memory.

*Trust, but verify.*

## Run it

Copy `SD-CARD/TINYBR61` to your SD card, keeping `TINYBR61.D81`, `MEGAQA.BIN` and `GLOW.BIN` together. In BASIC:

```basic
CHDIR "/",U12
CHDIR "TINYBR61",U12
MOUNT "TINYBR61.D81"
RUN "TINYBR61",U8
```

**RETURN:** ask · **F1:** next example · **DEL:** edit · **ESC:** stop/clear

Try `what is sid`, `how much attic ram is there` or `how do i load a program`.

## Source and training

The original model was trained from scratch on selected MEGA65 documentation converted into short question/answer pairs and varied phrasings. V6.1 fine-tunes our own V6 checkpoint, using quantization-aware training. The selected update took about **three minutes on an RTX 5090**, including checkpoint exports and evaluation; preparing data and testing took much longer.

The `source/` folder includes the model, training data and scripts, checkpoints, exporter, and native runtime. See [README.txt](README.txt) for build instructions, Xemu setup and detailed results. Tool paths need adjusting for your machine.

The architecture and inference engine grew out of the GoatSong/SIDLM experiment. Documentation comes from the [MEGA65 User Guide](https://github.com/MEGA65/mega65-user-guide). See [SOURCE-NOTICES.txt](SOURCE-NOTICES.txt) for attribution and licensing details; the included manual sources and derived training text use the [GFDL](GFDL-1.3.txt).
