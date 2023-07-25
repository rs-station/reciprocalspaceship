from reciprocalspaceship.dtypes.internals import NumericArray, NumericDtype


class MTZDtype(NumericDtype):
    """Base ExtensionDtype for implementing persistent MTZ data types"""

    def __repr__(self) -> str:
        return f"{self.name}"

    @property
    def _is_numeric(self) -> bool:
        return True

    @classmethod
    def construct_array_type(cls):
        """
        Return the array type associated with this dtype.

        Returns
        -------
        type
        """
        return NotImplementedError

    @classmethod
    def construct_from_string(cls, string):
        if not isinstance(string, str):
            raise TypeError(
                f"'construct_from_string' expects a string, got {type(string)}"
            )
        elif string != cls.name and string != cls.mtztype:
            raise TypeError(f"Cannot construct a '{cls.__name__}' from '{string}'")
        return cls()

    def is_friedel_dtype(self):
        """Returns whether MTZ dtype represents a Friedel dtype"""
        raise NotImplementedError
