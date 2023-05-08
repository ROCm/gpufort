
log_prefix = "linemapper" # prefix for logging
pretty_print_linemaps_dump = False
        # Prettify linemaps JSON files.
        #PATTERN_LINE_CONTINUATION=r"(\&\s*\n\s*([!c\*]\$\w+)?)|(^\s*[!c\*]\$\w+\&\s*)"
#pattern_line_continuation = r"\&([!c\*][@\$]\w+)?|([!c\*][@\$]\w+\&)"
        # line continuation pattern. The linemapper's preprocessor removes them.

        # 'strict': program terminates with error. Otherwise, a warning is printed.

modern_fortran = True
        # Expect modern Fortran comments and directives
comment_prefix_for_inactive_code = "! "
        # Inactive code within translated code regions gets commented out.
        # This is the prefix to use for comments.
cuda_fortran = True
        # Expect conditional code such as `!@cuf ierr = cudaDeviceSynchronize()`.
user_defined_macros = []
        # manually add macro definitions: dicts with entries 'name' (str),
        # 'args' (list of str), and 'subst' (str)

only_apply_user_defined_macros = False
        # Only apply user defined macros (incl. compiler options) and turn
        # off other preprocessing (-> all code is active)

indent_width_whitespace = 2
        # number of indent chars if indentation uses whitespaces
indent_width_tabs = 1
        # number of indent chars if indentation uses tabs

default_indent_char = ' '
        # The default index char to use if no other char was detected (' ' or '\t').

line_grouping_include_blank_lines = True
line_grouping_ifdef_macro = "__GPUFORT"
        # If not set to None, introduce ifdef-else-endif preprocessor block around modified lines
        # and keep the original in the else branch.
        # This is the to check in the ifdef directive.
