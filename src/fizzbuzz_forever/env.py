class FizzBuzzEnv:
    """Environment for fizzbuzz agent inference."""

    def modulo(self, a: int, b: int) -> int:
        """Compute the remainder of dividing two integers.

        Args:
            a: The dividend.
            b: The divisor.

        Returns:
            The remainder of a divided by b.
        """
        if b == 0:
            return 0
        return a % b
