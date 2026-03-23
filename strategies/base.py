from abc import ABC, abstractmethod
import pandas as pd


class BaseStrategy(ABC):

    NAME: str = ""
    VERSION: str = "1.0.0"
    AUTHOR: str = ""
    DESCRIPTION: str = ""

    def __init__(self, params: dict):
        self.params = params
        self._validate_params()

    @abstractmethod
    def _validate_params(self) -> None:
        pass

    @abstractmethod
    def generate_signals(
        self,
        prices: pd.DataFrame,
        mcap: pd.DataFrame,
        **kwargs,
    ) -> pd.DataFrame:
        """
        Returns DataFrame of position weights.
        Index: dates | Columns: tokens | Values: weight (-1 to 1)
        """
        pass

    def get_metadata(self) -> dict:
        return {
            "name":    self.NAME,
            "version": self.VERSION,
            "author":  self.AUTHOR,
            "params":  self.params,
        }
