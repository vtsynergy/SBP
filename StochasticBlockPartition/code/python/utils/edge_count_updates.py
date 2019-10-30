"""The edge count updates class.
"""

import numpy as np


class EdgeCountUpdates(object):
    """Holds the updates to the current interblock edge counts given a proposed block or node move.

    Since a block move affects only the rows and columns for the original and proposed blocks, only four rows and
    columns need to be stored for the edge count matrix updates.
    """

    def __init__(self, block_row: np.array, proposal_row: np.array, block_col: np.array,
                 proposal_col: np.array) -> None:
        """Creates a new EdgeCountUpdates object.

            Parameters
            ---------
            block_row : np.array [int]
                    the updates for the row of the current block
            proposal_row : np.array [int]
                    the updates for the row of the proposed block
            block_col : np.array [int]
                    the updates for the column of the current block
            proposal_col : np.array [int]
                    the updates for the column of the proposed block
        """
        self.block_row = block_row
        self.proposal_row = proposal_row
        self.block_col = block_col
        self.proposal_col = proposal_col
    # End of __init__()

    def __eq__(self, other: object):
        """Returns True if this EdgeCountUpdates object contains the same values as the other, False otherwise.
        """
        return (
            isinstance(other, EdgeCountUpdates) and
            np.array_equal(self.block_row, other.block_row) and
            np.array_equal(self.proposal_row, other.proposal_row) and
            np.array_equal(self.block_col, other.block_col) and
            np.array_equal(self.proposal_col, other.proposal_col)
        )
    # End of __eq__()
# End of EdgeCountUpdates()
