# FizzBuzz FOREVER - Agent Edition

Don't hurt your brain trying to figure out fizzbuzz, make an LLM do it!
Be done with fizzbuzz forevar.

1. Clone repo
2. Train an LLM
3. Hook up to a for loop
4. $$ profit $$

## Quick Start

LOL so easy as long as you have a Blackwell Pro RTX 6000.
If you don't, then you might need to muck around in the `config.yaml`
and reduce the batch size. Your interviewer will probably understand if you
need to take a minute to setup a RunPod instance.

```sh
uv sync
uv run fizzbuzz-train
```

> [!NOTE]
> This is gonna take a few minutes to train, providing a great opportunity to talk
> about how you could set this up with Agent2Agent protocol.

## Kick Ass Features

- Train with ultra efficient Qwen-0.6B model
- Fast fast fast inference with KV caching
- Statistically pretty darn accurate, even with less than an hour of training
- It uses tools so you know it's probably gonna be right
- Once you drop the phrase *fizzbuzz agentic workflow* on them, you're hired

## FizzBuzz ~~Simplified~~ Streamlined

```python
from fizzbuzz_forever.agent import FizzBuzzAgent

agent = FizzBuzzAgent.load('outputs/final_model')
for n in range(1, 101):
    print(agent(n))

# Example Output (probably)
# 1 1
# 2 2
# 3 Fizz
# 4 4
# 5 Buzz
# ...
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

## FAQ

### Can I use this in production?

Yes.


### Is this better than a for loop?

Are `for` loops agentic?
Do you want to show your interviewer some 80s shit or impress them?


### What if my interviewer asks about Big O notation?

Doesn't apply. We're using tensors here.
Tell them to get their Little O brain up to date with 2026.


### Why not just use ChatGPT?

Fair question. ChatGPT can probably get it right if you provide the tools.
However, you can deploy this on an `ml.g6.xlarge` instance for cost efficiency.



