# SPDX-License-Identifier: MIT
# Copyright (c) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
from . import base

from .. import error

class TokenStream():
    
    def __init__(self,tokens,**kwargs):
        self.setup(tokens,**kwargs)
    
    def setup(self,
          tokens,
          modern_fortran=True,
          keepws=True,
          ignorews=True,
          padded_size=0,
          throw_syntax_instead_of_index_error = True
        ):
        """
        :param ignorews: Ignore white space tokens when getting the next token.
        """
        if isinstance(tokens,TokenStream):
            self.tokens = tokens.tokens
            self.current_front = tokens.current_front
        elif isinstance(tokens,list):
            self.tokens = tokens
            self.current_front = 0
        elif isinstance(tokens,str):
            self.tokens = base.tokenize(tokens,
              padded_size,
              modern_fortran,
              keepws
            )
            self.current_front = 0
            self.ignorews = ignorews
        else:
            assert False, "unexpected input"
        self.ignorews = ignorews
        #
        self._throw_syntax_instead_of_index_error = throw_syntax_instead_of_index_error

    def _remaining_tokens_gen(self):
        """:return: Next tokens in the stream (search includes token at self.current_front)."""
        if self.ignorews:
            for rel_idx,tk in enumerate(self.tokens[self.current_front:]):
                if not base.is_blank(tk):
                    yield (rel_idx,tk)
        else:
            for rel_idx,tk in enumerate(self.tokens[self.current_front:]):
                yield (rel_idx,tk)

    def __getitem__(self, idx):
        if isinstance( idx, slice ):
            assert False, "slicing not implemented"
        elif isinstance( idx, int ):
            if self.ignorews:
                ctr: int = 0
                for _,tk in self._remaining_tokens_gen():
                    if ctr == idx:
                        return tk
                    ctr += 1
                if self._throw_syntax_instead_of_index_error:
                    raise error.SyntaxError("reached end of tokens")  
                else:
                    raise IndexError("reached end of tokens")  
            else:
                return self.tokens[self.current_front+idx]

    def size(self,countws=False):
        """:param bool countws: Force counting of whitespace tokens.
        """
        if self.ignorews and not countws:
            return len([tk for tk in self.tokens[self.current_front:] if not base.is_blank(tk)])
        else:
            return len(self.tokens[self.current_front:])
  
    def empty(self,countws=False):
        """:param bool countws: Force counting of whitespace tokens.
        """
        return self.size(countws) == 0
    
    def __len__(self):
        """:return: remaining lengths."""
        return self.size()
  
    def __str__(self):
        return (
          "TokenStream(tokens=" + str(self.tokens)
          + ",current_front=" + str(self.current_front)
          + ")"
        )
    __repr__ = __str__
    
    def pop_front(self,num_tokens=1):
        """:return: Next n tokens in the stream.
        :note: Increments self.current_front.
        :throw: util.error.SyntaxError if end of tokens is reached.
        """
        result: list[str] = []
        last_index: int = -1
        ctr: int = 0
        for rel_idx,tk in self._remaining_tokens_gen():
            if ctr >= num_tokens:
                last_index = rel_idx
                break
            else:
                result.append(tk)
                ctr += 1
        if len(result) < num_tokens:
            if self._throw_syntax_instead_of_index_error:
                raise error.SyntaxError("reached end of tokens")
            else:
                raise IndexError("of of range")
        if last_index >= 0:
            self.current_front = self.current_front + last_index
        else:
            self.current_front = len(self.tokens)
        if num_tokens == 1:
            return result[0]
        else:
            return result
    
    def pop_front_equals(self,*tokens):
        """:return: Next n tokens in the stream equal the tokens in 'other'
                    element per element.
        :note: Increments self.current_front.
        :throw: util.error.SyntaxError if end of tokens is reached.
        """
        return base.compare(self.pop_front(len(tokens)),tokens)
    
    def pop_front_equals_ignore_case(self,*tokens):
        """:return: Case-insensitive 'pop_front_equals'.
        :throw: util.error.SyntaxError if end of tokens is reached.
        """
        return base.compare_ignore_case(self.pop_front(len(tokens)),tokens)

    def check_if_remaining_tokens_are_blank(self):
        base.check_if_all_tokens_are_blank(self.tokens[self.current_front:])
