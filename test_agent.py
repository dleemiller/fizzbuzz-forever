import pytest
from fizzbuzz_forever.agent import FizzBuzzAgent


@pytest.fixture(scope="session")
def agent():
    return FizzBuzzAgent.load("outputs/final_model")


@pytest.mark.parametrize("n", range(1, 16))
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
