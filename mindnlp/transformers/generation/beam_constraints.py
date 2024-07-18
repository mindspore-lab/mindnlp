# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""
Beam constraints
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class Constraint(ABC):
    r"""Abstract base class for all constraints that can be applied during generation.
    It must define how the constraint can be satisfied.

    All classes that inherit Constraint must follow the requirement that

    Example:
        ```python
        >>> completed = False
        >>> while not completed:
        >>>     _, completed = constraint.update(constraint.advance())
        ```

    will always terminate (halt).
    """
    def __init__(self):
        """
        Initializes an instance of the Constraint class.

        Args:
            self: Constraint instance being initialized.

        Returns:
            None.

        Raises:
            None.
        """
        # test for the above condition
        self.test()

    def test(self):
        """
        Tests whether this constraint has been properly defined.
        """
        counter = 0
        completed = False
        while not completed:
            if counter == 1:
                self.reset()
            advance = self.advance()
            if not self.does_advance(advance):
                raise RuntimeError(
                    "Custom Constraint is not defined correctly. self.does_advance(self.advance()) must be true."
                )

            _, completed, _ = self.update(advance)
            counter += 1

            if counter > 10000:
                raise RuntimeError("update() does not fulfill the constraint.")

        if self.remaining() != 0:
            raise RuntimeError("Custom Constraint is not defined correctly.")

    @abstractmethod
    def advance(self):
        """
        When called, returns the token that would take this constraint one step closer to being fulfilled.

        Returns:
            token_ids(`torch.tensor`): Must be a tensor of a list of indexable tokens, not some integer.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def does_advance(self, token_id: int):
        """
        Reads in a token and returns whether it creates progress.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def update(self, token_id: int):
        """
        Reads in a token and returns booleans that indicate the progress made by it. This function will update the
        state of this object unlikes `does_advance(self, token_id: int)`.

        This isn't to test whether a certain token will advance the progress; it's to update its state as if it has
        been generated. This becomes important if token_id != desired token (refer to else statement in
        PhrasalConstraint)

        Args:
            token_id(`int`):
                The id of a newly generated token in the beam search.
        Returns:
            stepped(`bool`):
                Whether this constraint has become one step closer to being fulfuilled.
            completed(`bool`):
                Whether this constraint has been completely fulfilled by this token being generated.
            reset (`bool`):
                Whether this constraint has reset its progress by this token being generated.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def reset(self):
        """
        Resets the state of this constraint to its initialization. We would call this in cases where the fulfillment of
        a constraint is abrupted by an unwanted token.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def remaining(self):
        """
        Returns the number of remaining steps of `advance()` in order to complete this constraint.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )

    @abstractmethod
    def copy(self, stateful=False):
        """
        Creates a new instance of this constraint.

        Args:
            stateful(`bool`): Whether to not only copy the constraint for new instance, but also its state.

        Returns:
            constraint(`Constraint`): The same constraint as the one being called from.
        """
        raise NotImplementedError(
            f"{self.__class__} is an abstract class. Only classes inheriting this class can be called."
        )


class PhrasalConstraint(Constraint):
    r"""
    [`Constraint`] enforcing that an ordered sequence of tokens is included in the output.

    Args:
        token_ids (`List[int]`):
            The id of the token that must be generated by the output.
    """
    def __init__(self, token_ids: List[int]):
        """
        __init__

        Initializes a new instance of the PhrasalConstraint class.

        Args:
            self: The instance of the PhrasalConstraint class.
            token_ids (List[int]): A list of token IDs representing the constraints.
                This parameter is required and should be a non-empty list of positive integers.

        Returns:
            None.

        Raises:
            ValueError: If token_ids is not a non-empty list or if it contains non-positive integers.
        """
        super(Constraint, self).__init__()

        if not isinstance(token_ids, list) or len(token_ids) == 0:
            raise ValueError(f"`token_ids` has to be a non-empty list, but is {token_ids}.")
        if any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids):
            raise ValueError(f"Each list in `token_ids` has to be a list of positive integers, but is {token_ids}.")

        self.token_ids = token_ids

        self.seqlen = len(self.token_ids)
        self.fulfilled_idx = -1  # the index of the currently fulfilled step
        self.completed = False

    def advance(self):
        """Advance to the next token in the PhrasalConstraint.

        Args:
            self (PhrasalConstraint): The PhrasalConstraint instance.

        Returns:
            None: If the PhrasalConstraint is completed, returns None.
            int: The next token ID if the PhrasalConstraint is not completed.

        Raises:
            None.
        """
        if self.completed:
            return None
        return self.token_ids[self.fulfilled_idx + 1]

    def does_advance(self, token_id: int):
        """
        Checks if the given `token_id` can be advanced in the context of the PhrasalConstraint class.

        Args:
            self (PhrasalConstraint): An instance of the PhrasalConstraint class.
            token_id (int): The ID of the token to be advanced.

        Returns:
            None.

        Raises:
            ValueError: If the `token_id` parameter is not of type int.
        """
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        if self.completed:
            return False

        return token_id == self.token_ids[self.fulfilled_idx + 1]

    def update(self, token_id: int):
        """
        Updates the state of the PhrasalConstraint object based on the given token ID.

        Args:
            self (PhrasalConstraint): The PhrasalConstraint object.
            token_id (int): The ID of the token to update the state with.

        Returns:
            None.

        Raises:
            ValueError: If the `token_id` is not an integer.

        This method updates the state of the PhrasalConstraint object by either advancing the fulfillment index,
        marking the constraint as completed, or resetting the state. The method returns None.

        If the `token_id` is not an integer, a ValueError is raised with a descriptive error message.

        Note:
            The method modifies the state of the PhrasalConstraint object by updating the fulfillment index,
            completion status, and potentially resetting the state.
        """
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` has to be an `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.fulfilled_idx += 1
            stepped = True
            if self.fulfilled_idx == (self.seqlen - 1):
                completed = True
            self.completed = completed
        else:
            # failed to make progress.
            reset = True
            self.reset()
        return stepped, completed, reset

    def reset(self):
        """
        Resets the state of the PhrasalConstraint object.

        Args:
            self:
                PhrasalConstraint

                - The instance of the PhrasalConstraint class.
                Represents the current PhrasalConstraint object to be reset.

        Returns:
            None.

        Raises:
            None.
        """
        self.completed = False
        self.fulfilled_idx = 0

    def remaining(self):
        """
        This method calculates the remaining length of the sequence that needs to be fulfilled for the phrasal constraint.

        Args:
            self (PhrasalConstraint): The instance of the PhrasalConstraint class.

        Returns:
            int: The remaining length of the sequence to be fulfilled for the phrasal constraint.

        Raises:
            None
        """
        return self.seqlen - (self.fulfilled_idx + 1)

    def copy(self, stateful=False):
        """
        Copy a PhrasalConstraint.

        Args:
            self (PhrasalConstraint): The instance of the PhrasalConstraint class.
            stateful (bool): If True, the copy will include the stateful attributes of the constraint.
                Defaults to False.

        Returns:
            PhrasalConstraint: A new instance of the PhrasalConstraint class with a copy of the token_ids.
                If stateful is True, the new instance will also have the same seq_len, fulfilled_idx,
                and completed attributes as the original instance.

        Raises:
            None.
        """
        new_constraint = PhrasalConstraint(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.fulfilled_idx = self.fulfilled_idx
            new_constraint.completed = self.completed

        return new_constraint


class DisjunctiveTrie:
    """DisjunctiveTrie"""
    def __init__(self, nested_token_ids: List[List[int]], no_subsets=True):
        r"""
        A helper class that builds a trie with the words represented in `nested_token_ids`.
        """
        self.max_height = max(len(one) for one in nested_token_ids)

        root = {}
        for token_ids in nested_token_ids:
            level = root
            for _, token_id in enumerate(token_ids):
                if token_id not in level:
                    level[token_id] = {}

                level = level[token_id]

        if no_subsets and self.has_subsets(root, nested_token_ids):
            raise ValueError(
                "Each list in `nested_token_ids` can't be a complete subset of another list, but is"
                f" {nested_token_ids}."
            )

        self.trie = root

    def next_tokens(self, current_seq):
        """
        The next possible tokens that will progress the trie, given the current sequence of tokens in `current_seq`.
        """
        start = self.trie

        for current_token in current_seq:
            start = start[current_token]

        next_tokens = list(start.keys())

        return next_tokens

    def reached_leaf(self, current_seq):
        """
        This method is part of the DisjunctiveTrie class and is used to determine if the current sequence has reached
        a leaf node within the trie structure.

        Args:
            self: The instance of the DisjunctiveTrie class.
            current_seq: A sequence representing the current state within the trie. It is of type str and is used to
                navigate through the trie structure. There are no specific restrictions on the content of the sequence.

        Returns:
            None: This method returns a value of type None, indicating that there are no more tokens to traverse in the
                trie, and the current sequence has reached a leaf node.

        Raises:
            None.
        """
        next_tokens = self.next_tokens(current_seq)

        return len(next_tokens) == 0

    def count_leaves(self, root):
        """
        Counts the number of leaves in a Disjunctive Trie starting from a given root node.

        Args:
            self (DisjunctiveTrie): The instance of the DisjunctiveTrie class.
            root (dict): The root node of the Disjunctive Trie from which the leaf count should be calculated.

        Returns:
            None.

        Raises:
            None.
        """
        next_nodes = list(root.values())
        if len(next_nodes) == 0:
            return 1
        return sum(self.count_leaves(nn) for nn in next_nodes)

    def has_subsets(self, trie, nested_token_ids):
        """
        Returns whether # of leaves == # of words. Otherwise some word is a subset of another.
        """
        leaf_count = self.count_leaves(trie)
        return len(nested_token_ids) != leaf_count


class DisjunctiveConstraint(Constraint):
    r"""
    A special [`Constraint`] that is fulfilled by fulfilling just one of several constraints.

    Args:
        nested_token_ids (`List[List[int]]`):
            A list of words, where each word is a list of ids. This constraint is fulfilled by generating just one from
            the list of words.
    """
    def __init__(self, nested_token_ids: List[List[int]]):
        """
        Initialize a DisjunctiveConstraint object with the provided nested_token_ids.

        Args:
            self: The instance of the DisjunctiveConstraint class.
            nested_token_ids (List[List[int]]): A list of lists containing positive integers representing token IDs.
                This parameter is required and must be a non-empty list of lists. Each inner list represents a
                sequence of token IDs. Each token ID should be a positive integer (greater than or equal to 0).

        Returns:
            None.

        Raises:
            ValueError: If nested_token_ids is not a list or is an empty list.
            ValueError: If nested_token_ids is not a list of lists.
            ValueError: If any inner list in nested_token_ids contains non-integer values or negative integers.

        """
        super(Constraint, self).__init__()

        if not isinstance(nested_token_ids, list) or len(nested_token_ids) == 0:
            raise ValueError(f"`nested_token_ids` has to be a non-empty list, but is {nested_token_ids}.")
        if any(not isinstance(token_ids, list) for token_ids in nested_token_ids):
            raise ValueError(f"`nested_token_ids` has to be a list of lists, but is {nested_token_ids}.")
        if any(
            any((not isinstance(token_id, int) or token_id < 0) for token_id in token_ids)
            for token_ids in nested_token_ids
        ):
            raise ValueError(
                f"Each list in `nested_token_ids` has to be a list of positive integers, but is {nested_token_ids}."
            )

        self.trie = DisjunctiveTrie(nested_token_ids)
        self.token_ids = nested_token_ids

        self.seqlen = self.trie.max_height
        self.current_seq = []
        self.completed = False

    def advance(self):
        """
        Advance the current sequence in the DisjunctiveConstraint object and return the next available token.

        Args:
            self (DisjunctiveConstraint): The current instance of the DisjunctiveConstraint class.

        Returns:
            None: If there are no more tokens available in the current sequence.

        Raises:
            None.

        """
        token_list = self.trie.next_tokens(self.current_seq)

        if len(token_list) == 0:
            return None
        return token_list

    def does_advance(self, token_id: int):
        """
        Checks if a given token ID can be advanced in the DisjunctiveConstraint.

        Args:
            self (DisjunctiveConstraint): The instance of the DisjunctiveConstraint class.
            token_id (int): The ID of the token to be checked for advancement.

        Returns:
            None: This method does not return any value. It only performs a check.

        Raises:
            ValueError: If the provided `token_id` is not of type int.

        Note:
            The `does_advance` method checks if the given `token_id` can be advanced in the DisjunctiveConstraint.
            It first validates that the `token_id` is of type int. Then, it retrieves the next possible tokens from the
            trie associated with the current sequence. Finally, it returns whether the `token_id` is present in the next
            tokens or not.
        """
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        next_tokens = self.trie.next_tokens(self.current_seq)

        return token_id in next_tokens

    def update(self, token_id: int):
        """
        This method updates the state of the DisjunctiveConstraint object based on the provided token_id.

        Args:
            self (DisjunctiveConstraint): The instance of the DisjunctiveConstraint class.
            token_id (int): The identifier of the token to be processed. It must be of type 'int'.

        Returns:
            None.

        Raises:
            ValueError: If the token_id provided is not of type 'int'.
        """
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` is supposed to be type `int`, but is {token_id} of type {type(token_id)}")

        stepped = False
        completed = False
        reset = False

        if self.does_advance(token_id):
            self.current_seq.append(token_id)
            stepped = True
        else:
            reset = True
            self.reset()

        completed = self.trie.reached_leaf(self.current_seq)
        self.completed = completed

        return stepped, completed, reset

    def reset(self):
        """
        Resets the state of the DisjunctiveConstraint.

        Args:
            self: The instance of the DisjunctiveConstraint class.

        Returns:
            None.

        Raises:
            None.
        """
        self.completed = False
        self.current_seq = []

    def remaining(self):
        """
        Returns the remaining length of the current sequence in a DisjunctiveConstraint object.

        Args:
            self (DisjunctiveConstraint): The instance of the DisjunctiveConstraint class.

        Returns:
            None.

        Raises:
            None.
        """
        if self.completed:
            # since this can be completed without reaching max height
            return 0
        return self.seqlen - len(self.current_seq)

    def copy(self, stateful=False):
        """
        Copy method creates a new instance of DisjunctiveConstraint and returns it. This method can be used to create
        a copy of an existing DisjunctiveConstraint object.

        Args:
            self (DisjunctiveConstraint): The current instance of the DisjunctiveConstraint object.
            stateful (bool): A flag indicating whether to create a stateful copy or not. If set to True,
                the state of the current instance will be copied to the new instance. Default is False.

        Returns:
            DisjunctiveConstraint: A new instance of the DisjunctiveConstraint object.

        Raises:
            None.

        Note:
            - If stateful is set to True, the new instance will have the same values for seq_len, current_seq,
            and completed as the current instance.
            - If stateful is set to False, the new instance will have default values for seq_len, current_seq,
            and completed.

        Example:
            ```python
            >>> constraint = DisjunctiveConstraint(['A', 'B', 'C'])
            >>> constraint.seq_len = 10
            >>> constraint.current_seq = ['A', 'B']
            >>> constraint.completed = False
            ...
            >>> # Create a stateful copy
            >>> new_constraint = constraint.copy(stateful=True)
            >>> # new_constraint.seq_len = 10
            >>> # new_constraint.current_seq = ['A', 'B']
            >>> # new_constraint.completed = False
            ...
            >>> # Create a non-stateful copy
            >>> new_constraint = constraint.copy(stateful=False)
            >>> # new_constraint.seq_len = default value
            >>> # new_constraint.current_seq = default value
            >>> # new_constraint.completed = default value
            ```
        """
        new_constraint = DisjunctiveConstraint(self.token_ids)

        if stateful:
            new_constraint.seq_len = self.seqlen
            new_constraint.current_seq = self.current_seq
            new_constraint.completed = self.completed

        return new_constraint


class ConstraintListState:
    r"""
    A class for beam scorers to track its progress through a list of constraints.

    Args:
        constraints (`List[Constraint]`):
            A list of [`Constraint`] objects that must be fulfilled by the beam scorer.
    """
    def __init__(self, constraints: List[Constraint]):
        """Initialize a ConstraintListState object.

        Args:
            self (ConstraintListState): The instance of the ConstraintListState class.
            constraints (List[Constraint]): A list of Constraint objects representing the constraints.

        Returns:
            None.

        Raises:
            None.
        """
        self.constraints = constraints

        # max # of steps required to fulfill a given constraint
        self.max_seqlen = max(c.seqlen for c in constraints)
        self.n_constraints = len(constraints)
        self.completed = False

        self.init_state()

    def init_state(self):
        """
        This method initializes the state of the ConstraintListState object.

        Args:
            self: ConstraintListState - The instance of the ConstraintListState class.

        Returns:
            None.

        Raises:
            None
        """
        self.complete_constraints = []
        self.inprogress_constraint = None
        self.pending_constraints = [constraint.copy(stateful=False) for constraint in self.constraints]

    def get_bank(self):
        """
        This method 'get_bank' is defined within the 'ConstraintListState' class and retrieves the bank value based on
        certain constraints.

        Args:
            self:
                An instance of the 'ConstraintListState' class.

                - Type: object
                - Purpose: Represents the current instance of the class.
                - Restrictions: None

        Returns:
            bank value: The method calculates and returns the bank value based on the complete and in-progress
                constraints as well as the maximum sequence length.

        Raises:
            None.
        """
        add = 0
        if self.inprogress_constraint:
            # extra points for having a constraint mid-fulfilled
            add += self.max_seqlen - self.inprogress_constraint.remaining()

        return (len(self.complete_constraints) * self.max_seqlen) + add

    def advance(self):
        """
        The list of tokens to generate such that we can make progress.
        By "list" we don't mean the list of token that will fully fulfill a constraint.

        Given constraints `c_i = {t_ij | j == # of tokens}`, If we're not in the middle of progressing through a
        specific constraint `c_i`, we return:

        `[t_k1 for k in indices of unfulfilled constraints]`

        If we are in the middle of a constraint, then we return:
            `[t_ij]`, where `i` is the index of the inprogress constraint, `j` is the next step for the constraint.

        Though we don't care which constraint is fulfilled first, if we are in the progress of fulfilling a constraint,
        that's the only one we'll return.
        """
        token_list = []
        if self.inprogress_constraint is None:
            for constraint in self.pending_constraints:  # "pending" == "unfulfilled yet"
                advance = constraint.advance()
                if isinstance(advance, int):
                    token_list.append(advance)
                elif isinstance(advance, list):
                    token_list.extend(advance)
        else:
            advance = self.inprogress_constraint.advance()
            if isinstance(advance, int):
                token_list.append(advance)
            elif isinstance(advance, list):
                token_list.extend(advance)

        if len(token_list) == 0:
            return None
        return token_list

    def reset(self, token_ids: Optional[List[int]]):
        """
        token_ids: the tokens generated thus far to reset the state of the progress through constraints.
        """
        self.init_state()

        if token_ids is not None:
            for token in token_ids:
                # completes or steps **one** constraint
                _, _ = self.add(token)

                # the entire list of constraints are fulfilled
                if self.completed:
                    break

    def add(self, token_id: int):
        """
        This method 'add' belongs to the class 'ConstraintListState' and is used to update the state based on the
        provided token_id.

        Args:
            self:
                Represents the instance of the 'ConstraintListState' class.

                - Type: ConstraintListState
                - Purpose: Allows access to the attributes and methods of the class instance.
                - Restrictions: None

            token_id:
                Represents the token identifier that needs to be processed.

                - Type: int
                - Purpose: Specifies the token identifier to be processed within the constraints.
                - Restrictions: Must be of integer type.

        Returns:
            tuple:
                The method returns a tuple containing two boolean values, 'complete' and 'stepped'.

                - Type: Tuple (bool, bool)
                - Purpose:

                    - 'complete': Indicates whether the state update operation is complete.
                    - 'stepped': Indicates whether any incremental progress was made during the update.

                - Restrictions: None
        
        Raises:
            ValueError: Raised when the 'token_id' parameter is not of integer type.
            RuntimeError: Raised when the update operation does not yield incremental progress
                despite the advancement condition being met.
        """
        if not isinstance(token_id, int):
            raise ValueError(f"`token_id` should be an `int`, but is `{token_id}`.")

        complete, stepped = False, False

        if self.completed:
            complete = True
            stepped = False
            return complete, stepped

        if self.inprogress_constraint is not None:
            # In the middle of fulfilling a constraint. If the `token_id` *does* makes an incremental progress to current
            # job, simply update the state

            stepped, complete, reset = self.inprogress_constraint.update(token_id)
            if reset:
                # 1. If the next token breaks the progress, then we must restart.
                #     e.g. constraint = "I love pies" and sequence so far is "I love" but `token_id` == "books".

                #     But that doesn't mean we self.init_state(), since we only reset the state for this particular
                #     constraint, not the full list of constraints.

                self.pending_constraints.append(self.inprogress_constraint.copy(stateful=False))
                self.inprogress_constraint = None

            if complete:
                # 2. If the next token completes the constraint, move it to completed list, set
                #     inprogress to None. If there are no pending constraints either, then this full list of constraints
                #     is complete.

                self.complete_constraints.append(self.inprogress_constraint)
                self.inprogress_constraint = None

                if len(self.pending_constraints) == 0:
                    # we're done!
                    self.completed = True

        else:
            # Not in the middle of fulfilling a constraint. So does this `token_id` helps us step towards any of our list
            # of constraints?

            for cidx, pending_constraint in enumerate(self.pending_constraints):
                if pending_constraint.does_advance(token_id):
                    stepped, complete, reset = pending_constraint.update(token_id)

                    if not stepped:
                        raise RuntimeError(
                            "`constraint.update(token_id)` is not yielding incremental progress, "
                            "even though `constraint.does_advance(token_id)` is true."
                        )

                    if complete:
                        self.complete_constraints.append(pending_constraint)
                        self.inprogress_constraint = None

                    if not complete and stepped:
                        self.inprogress_constraint = pending_constraint

                    if complete or stepped:
                        # If we made any progress at all, then it's at least not a "pending constraint".

                        self.pending_constraints = (
                            self.pending_constraints[:cidx] + self.pending_constraints[cidx + 1 :]
                        )

                        if len(self.pending_constraints) == 0 and self.inprogress_constraint is None:
                            # If there's no longer any pending after this and no inprogress either, then we must be
                            # complete.

                            self.completed = True

                        break  # prevent accidentally stepping through multiple constraints with just one token.

        return complete, stepped

    def copy(self, stateful=True):
        """
        This method creates a copy of the ConstraintListState object with the option to include stateful constraints.
        
        Args:
            self (ConstraintListState): The current instance of the ConstraintListState class.
            stateful (bool): A flag indicating whether to include stateful constraints in the copy. Defaults to True.
                If set to True, the copy will include complete_constraints and inprogress_constraint.
        
        Returns:
            ConstraintListState:
                A new instance of the ConstraintListState class with copied constraints based on the specified
                stateful parameter.
        
        Raises:
            None.
        """
        new_state = ConstraintListState(self.constraints)  # we actually never though self.constraints objects
        # throughout this process. So it's at initialization state.

        if stateful:
            new_state.complete_constraints = [
                constraint.copy(stateful=True) for constraint in self.complete_constraints
            ]
            if self.inprogress_constraint is not None:
                new_state.inprogress_constraint = self.inprogress_constraint.copy(stateful=True)
            new_state.pending_constraints = [constraint.copy() for constraint in self.pending_constraints]

        return new_state
