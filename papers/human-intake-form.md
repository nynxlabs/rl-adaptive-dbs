# Human intake form

For papers whose copy may not go to AI tools (`rights.rules.ai_tools: never`). A person reads the
paper and fills this in; an agent then turns each line into a pawtrace spec item in
`papers/<paper>/spec.yaml`, keeping your words and your citation, and asks you through a ledger
question when something is unclear. Nothing here is pre-filled from the paper.

## How to fill it in

- One line per detail. Add as many lines as you need; delete examples you don't use.
- **where:** the place in the paper — `§III.B`, `Eq. 4`, `Alg. 1 line 7`, `Table II`, `Fig. 3 caption`.
- **says:** one of
  - `yes` — the paper states it
  - `no` — we need it and the paper doesn't say
  - `unclear` — two readings fit (write both)
  - `conflict` — two places disagree (give both places)
- **value:** the number or setting, with units, if the paper gives one.
- **words:** your own words, or a short quote in quotation marks. Keep quotes short.
- Leave a section empty if the paper has nothing on it. Write `?` anywhere you're not sure; the
  agent will ask.

Line format:

```
- <what> — where: — says: — value: — words:
```

## 0. The paper

- title:
- authors:
- year, venue:
- DOI:
- which copy you read (published version, date downloaded):

## 1. The simulated circuit

The brain model the controller is tested on: which model, any changes the paper makes, how long
each control step simulates, which signal is measured and how.

-

## 2. What the controller sees

Its input at each step: which signal, window, preprocessing, scaling.

-

## 3. What the controller does

Which stimulation settings it controls, their ranges and step sizes, and how an action becomes
pulses.

-

## 4. Reward

The reward or cost, every term and constant.

-

## 5. Networks

Each network: what it's for, layers in order, sizes, activations, anything unusual.

-

## 6. Training

Algorithm and update rule, every hyperparameter, exploration, episodes and episode length,
seeds, when training starts and stops.

-

## 7. Testing / evaluation

How a trained controller is run for the results: length, starting state, seeds, what is
averaged.

-

## 8. Anything else the results depend on

Steps applied to the trained controller, hardware assumptions, baselines the paper compares
against.

-

## 9. Figures and tables to rebuild

One block per panel or table.

```
### Fig. <n><letter>
- shows:
- x axis (name, units, range):
- y axis (name, units, range):
- each line / bar / box:
- what it claims:
```

## 10. Numbers the text claims

Results stated in the text, e.g. "reduces X by 40%".

-

## 11. Questions for us

Anything you'd like checked, or that the agent should know.

-
