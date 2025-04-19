import numpy as np
from .linear_regression_interface import LinearRegression


class ClosedFormLinearRegression(LinearRegression):
    def fit(self, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
        """
        Linear regression implementation using the closed-form least squares solution.

        Covariance of x and y

        - Imagine plotting house‑size vs. house‑price as a cloud of points.
        Whenever a point lies to the right of average size(x > x̄) and above average price(y > ȳ), it "pulls" the line up
        likewise, points left and below pull it down.
        - Covariance is just a tally of all those "pulls" – every time size and price deviate from their means in the same direction, you add a positive contribution
        opposite directions give a negative contribution.
        - Larger positive covariance ⇒ on average big houses cost more.

        Variance of x
          - Now look just at the sizes. Some houses are much bigger than average, others much smaller.
          - Variance measures how spread‑out the sizes are around the mean.
          - If sizes barely vary, even a small change in price per square foot gives a big swing in predicted price
            if sizes vary a lot, the same price‑per‑sqft change is "diluted" over a wider range.

        Why slope = covariance / variance

        Slope is basically "how many dollars per square foot."
        Covariance tells you how strongly price moves when size moves.
        Variance tells you how big those size‑moves are by themselves.
        Dividing the two answers "for each unit of size‑spread, how much price‑spread do we get?"
        That ratio is exactly the line that, overall, balances all the upward and downward pulls so that the squared errors (vertical misses) are as small as possible.
        """
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        numerator = np.sum((x - x_mean) * (y - y_mean))
        denominator = np.sum((x - x_mean) ** 2)
        w = numerator / denominator
        b = y_mean - w * x_mean
        return w, b