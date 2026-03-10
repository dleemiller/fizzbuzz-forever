# FizzBuzz FOREVER - Agent Edition

Don't hurt your brain trying to figure out fizzbuzz, make an LLM do it!
Be done with fizzbuzz forevar.

Step 1: clone repo
Step 2: train an LLM
Step 3: hook up to a for loop
Step 4: $$ profit $$

## Quick Start

LOL so easy as long as you have an Blackwell Pro RTX 6000.
If you don't, then you might need to muck around in the `config.yaml`
and reduce the batch size. Your interviewer will probably understand if you
need to take a minute to setup a RunPod instance.

```
uv sync
uv run fizzbuzz-train
```

> [!NOTE]
> This is gonna take a few minutes to train, providing a great opportunity to talk
> about how you could set this up with Agent2Agent protocol.


## Kick Ass Features

- train with ultra efficient Qwen-0.6B model
- fast fast fast inference with KV caching
- statistically pretty darn accurate, even with less than an hour of training
- it uses tools so you know it's probably gonna be right
- once you drop the phrase *fizzbuzz agentic workflow* on them, you're hired

## FizzBuzz ~~Simplified~~ Streamlined

```
uv run python -c "
from fizzbuzz_forever.agent import FizzBuzzAgent
agent = FizzBuzzAgent.load('outputs/final_model')
for n in range(1, 101):
    print(agent(n))
"
```

## Robust Testing

It's probably going to work, so you'll totally get hired. But sometimes the interviewer
is going to ask you about testing. In the real world, nobody does that, except for Claude.
But just in case, try to memorize this test code. Fortunately, the logic to check
your outputs is simple:

```python
import pytest
from fizzbuzz_forever.agent import FizzBuzzAgent


@pytest.fixture(scope="session")
def agent():
    return FizzBuzzAgent.load("outputs/final_model")


@pytest.mark.parametrize("n", range(1, 101))
def test_fizzbuzz(agent, n):
    result = agent(n)
    number, label = result.split(" ", 1)
    assert int(number) == n
    if n % 15 == 0:
        assert label == "FizzBuzz"
    elif n % 3 == 0:
        assert label == "Fizz"
    elif n % 5 == 0:
        assert label == "Buzz"
    else:
        assert label == str(n)
```


### What If My Test Fails?

It's going to pass most of the time, so you're probably fine.
But if you're trying to deploy this in a CI/CD pipeline, it might fail on
occasion. Totally fine, one issue is testing *too much*. Just reduce the
parameterized range.

Otherwise, just tell your devops person to keep running it until it passes.
That's their job anyway.


