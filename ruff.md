
Skip to content
logo
Ruff
Rules
Type to start searching
ruff
Ruff

    Overview
    Tutorial
    Installation
    Usage
    Configuration
    Preview
    Rules
        Pyflakes (F)
        pycodestyle (E, W)
            Warning (W)
            Error (E)
        mccabe (C90)
        isort (I)
        pep8-naming (N)
        pydocstyle (D)
        pyupgrade (UP)
        flake8-2020 (YTT)
        flake8-annotations (ANN)
        flake8-async (ASYNC)
        flake8-bandit (S)
        flake8-blind-except (BLE)
        flake8-boolean-trap (FBT)
        flake8-bugbear (B)
        flake8-builtins (A)
        flake8-commas (COM)
        flake8-copyright (CPY)
        flake8-comprehensions (C4)
        flake8-datetimez (DTZ)
        flake8-debugger (T10)
        flake8-django (DJ)
        flake8-errmsg (EM)
        flake8-executable (EXE)
        flake8-future-annotations (FA)
        flake8-implicit-str-concat (ISC)
        flake8-import-conventions (ICN)
        flake8-logging-format (G)
        flake8-no-pep420 (INP)
        flake8-pie (PIE)
        flake8-print (T20)
        flake8-pyi (PYI)
        flake8-pytest-style (PT)
        flake8-quotes (Q)
        flake8-raise (RSE)
        flake8-return (RET)
        flake8-self (SLF)
        flake8-slots (SLOT)
        flake8-simplify (SIM)
        flake8-tidy-imports (TID)
        flake8-type-checking (TCH)
        flake8-gettext (INT)
        flake8-unused-arguments (ARG)
        flake8-use-pathlib (PTH)
        flake8-todos (TD)
        flake8-fixme (FIX)
        eradicate (ERA)
        pandas-vet (PD)
        pygrep-hooks (PGH)
        Pylint (PL)
        tryceratops (TRY)
        flynt (FLY)
        NumPy-specific rules (NPY)
        Airflow (AIR)
        Perflint (PERF)
        refurb (FURB)
        flake8-logging (LOG)
        Ruff-specific rules (RUF)
    Settings
    Editor Integrations
    FAQ
    Contributing

Rules#

Ruff supports over 700 lint rules, many of which are inspired by popular tools like Flake8, isort, pyupgrade, and others. Regardless of the rule's origin, Ruff re-implements every rule in Rust as a first-party feature.

By default, Ruff enables Flake8's E and F rules. Ruff supports all rules from the F category, and a subset of the E category, omitting those stylistic rules made obsolete by the use of an autoformatter, like Black.

If you're just getting started with Ruff, the default rule set is a great place to start: it catches a wide variety of common errors (like unused imports) with zero configuration.

The 🛠️ emoji indicates that a rule is automatically fixable by the --fix command-line option.

The 🧪 emoji indicates that a rule in "preview".
Pyflakes (F)#

For more, see Pyflakes on PyPI.
Code 	Name 	Message 	
F401 	unused-import 	{name} imported but unused; consider using importlib.util.find_spec to test for availability 	🛠️ 🧪
F402 	import-shadowed-by-loop-var 	Import {name} from line {line} shadowed by loop variable 	🛠️ 🧪
F403 	undefined-local-with-import-star 	from {name} import * used; unable to detect undefined names 	🛠️ 🧪
F404 	late-future-import 	from __future__ imports must occur at the beginning of the file 	🛠️ 🧪
F405 	undefined-local-with-import-star-usage 	{name} may be undefined, or defined from star imports 	🛠️ 🧪
F406 	undefined-local-with-nested-import-star-usage 	from {name} import * only allowed at module level 	🛠️ 🧪
F407 	future-feature-not-defined 	Future feature {name} is not defined 	🛠️ 🧪
F501 	percent-format-invalid-format 	%-format string has invalid format string: 	🛠️ 🧪
F502 	percent-format-expected-mapping 	%-format string expected mapping but got sequence 	🛠️ 🧪
F503 	percent-format-expected-sequence 	%-format string expected sequence but got mapping 	🛠️ 🧪
F504 	percent-format-extra-named-arguments 	%-format string has unused named argument(s): 	🛠️ 🧪
F505 	percent-format-missing-argument 	%-format string is missing argument(s) for placeholder(s): 	🛠️ 🧪
F506 	percent-format-mixed-positional-and-named 	%-format string has mixed positional and named placeholders 	🛠️ 🧪
F507 	percent-format-positional-count-mismatch 	%-format string has {wanted} placeholder(s) but {got} substitution(s) 	🛠️ 🧪
F508 	percent-format-star-requires-sequence 	%-format string * specifier requires sequence 	🛠️ 🧪
F509 	percent-format-unsupported-format-character 	%-format string has unsupported format character {char} 	🛠️ 🧪
F521 	string-dot-format-invalid-format 	.format call has invalid format string: 	🛠️ 🧪
F522 	string-dot-format-extra-named-arguments 	.format call has unused named argument(s): 	🛠️ 🧪
F523 	string-dot-format-extra-positional-arguments 	.format call has unused arguments at position(s): 	🛠️ 🧪
F524 	string-dot-format-missing-arguments 	.format call is missing argument(s) for placeholder(s): 	🛠️ 🧪
F525 	string-dot-format-mixing-automatic 	.format string mixes automatic and manual numbering 	🛠️ 🧪
F541 	f-string-missing-placeholders 	f-string without any placeholders 	🛠️ 🧪
F601 	multi-value-repeated-key-literal 	Dictionary key literal {name} repeated 	🛠️ 🧪
F602 	multi-value-repeated-key-variable 	Dictionary key {name} repeated 	🛠️ 🧪
F621 	expressions-in-star-assignment 	Too many expressions in star-unpacking assignment 	🛠️ 🧪
F622 	multiple-starred-expressions 	Two starred expressions in assignment 	🛠️ 🧪
F631 	assert-tuple 	Assert test is a non-empty tuple, which is always True 	🛠️ 🧪
F632 	is-literal 	Use == to compare constant literals 	🛠️ 🧪
F633 	invalid-print-syntax 	Use of >> is invalid with print function 	🛠️ 🧪
F634 	if-tuple 	If test is a tuple, which is always True 	🛠️ 🧪
F701 	break-outside-loop 	break outside loop 	🛠️ 🧪
F702 	continue-outside-loop 	continue not properly in loop 	🛠️ 🧪
F704 	yield-outside-function 	{keyword} statement outside of a function 	🛠️ 🧪
F706 	return-outside-function 	return statement outside of a function/method 	🛠️ 🧪
F707 	default-except-not-last 	An except block as not the last exception handler 	🛠️ 🧪
F722 	forward-annotation-syntax-error 	Syntax error in forward annotation: {body} 	🛠️ 🧪
F811 	redefined-while-unused 	Redefinition of unused {name} from line 	🛠️ 🧪
F821 	undefined-name 	Undefined name {name} 	🛠️ 🧪
F822 	undefined-export 	Undefined name {name} in __all__ 	🛠️ 🧪
F823 	undefined-local 	Local variable {name} referenced before assignment 	🛠️ 🧪
F841 	unused-variable 	Local variable {name} is assigned to but never used 	🛠️ 🧪
F842 	unused-annotation 	Local variable {name} is annotated but never used 	🛠️ 🧪
F901 	raise-not-implemented 	raise NotImplemented should be raise NotImplementedError 	🛠️ 🧪
pycodestyle (E, W)#

For more, see pycodestyle on PyPI.

For related settings, see pycodestyle.
Warning (W)#
Code 	Name 	Message 	
W191 	tab-indentation 	Indentation contains tabs 	🛠️ 🧪
W291 	trailing-whitespace 	Trailing whitespace 	🛠️ 🧪
W292 	missing-newline-at-end-of-file 	No newline at end of file 	🛠️ 🧪
W293 	blank-line-with-whitespace 	Blank line contains whitespace 	🛠️ 🧪
W505 	doc-line-too-long 	Doc line too long ({width} > {limit} characters) 	🛠️ 🧪
W605 	invalid-escape-sequence 	Invalid escape sequence: \{char} 	🛠️ 🧪
Error (E)#
Code 	Name 	Message 	
E101 	mixed-spaces-and-tabs 	Indentation contains mixed spaces and tabs 	🛠️ 🧪
E111 	indentation-with-invalid-multiple 	Indentation is not a multiple of 	🛠️ 🧪
E112 	no-indented-block 	Expected an indented block 	🛠️ 🧪
E113 	unexpected-indentation 	Unexpected indentation 	🛠️ 🧪
E114 	indentation-with-invalid-multiple-comment 	Indentation is not a multiple of {indent_size} (comment) 	🛠️ 🧪
E115 	no-indented-block-comment 	Expected an indented block (comment) 	🛠️ 🧪
E116 	unexpected-indentation-comment 	Unexpected indentation (comment) 	🛠️ 🧪
E117 	over-indented 	Over-indented (comment) 	🛠️ 🧪
E201 	whitespace-after-open-bracket 	Whitespace after '{symbol}' 	🛠️ 🧪
E202 	whitespace-before-close-bracket 	Whitespace before '{symbol}' 	🛠️ 🧪
E203 	whitespace-before-punctuation 	Whitespace before '{symbol}' 	🛠️ 🧪
E211 	whitespace-before-parameters 	Whitespace before '{bracket}' 	🛠️ 🧪
E221 	multiple-spaces-before-operator 	Multiple spaces before operator 	🛠️ 🧪
E222 	multiple-spaces-after-operator 	Multiple spaces after operator 	🛠️ 🧪
E223 	tab-before-operator 	Tab before operator 	🛠️ 🧪
E224 	tab-after-operator 	Tab after operator 	🛠️ 🧪
E225 	missing-whitespace-around-operator 	Missing whitespace around operator 	🛠️ 🧪
E226 	missing-whitespace-around-arithmetic-operator 	Missing whitespace around arithmetic operator 	🛠️ 🧪
E227 	missing-whitespace-around-bitwise-or-shift-operator 	Missing whitespace around bitwise or shift operator 	🛠️ 🧪
E228 	missing-whitespace-around-modulo-operator 	Missing whitespace around modulo operator 	🛠️ 🧪
E231 	missing-whitespace 	Missing whitespace after '{token}' 	🛠️ 🧪
E241 	multiple-spaces-after-comma 	Multiple spaces after comma 	🛠️ 🧪
E242 	tab-after-comma 	Tab after comma 	🛠️ 🧪
E251 	unexpected-spaces-around-keyword-parameter-equals 	Unexpected spaces around keyword / parameter equals 	🛠️ 🧪
E252 	missing-whitespace-around-parameter-equals 	Missing whitespace around parameter equals 	🛠️ 🧪
E261 	too-few-spaces-before-inline-comment 	Insert at least two spaces before an inline comment 	🛠️ 🧪
E262 	no-space-after-inline-comment 	Inline comment should start with # 	🛠️ 🧪
E265 	no-space-after-block-comment 	Block comment should start with # 	🛠️ 🧪
E266 	multiple-leading-hashes-for-block-comment 	Too many leading # before block comment 	🛠️ 🧪
E271 	multiple-spaces-after-keyword 	Multiple spaces after keyword 	🛠️ 🧪
E272 	multiple-spaces-before-keyword 	Multiple spaces before keyword 	🛠️ 🧪
E273 	tab-after-keyword 	Tab after keyword 	🛠️ 🧪
E274 	tab-before-keyword 	Tab before keyword 	🛠️ 🧪
E275 	missing-whitespace-after-keyword 	Missing whitespace after keyword 	🛠️ 🧪
E401 	multiple-imports-on-one-line 	Multiple imports on one line 	🛠️ 🧪
E402 	module-import-not-at-top-of-file 	Module level import not at top of file 	🛠️ 🧪
E501 	line-too-long 	Line too long ({width} > {limit} characters) 	🛠️ 🧪
E701 	multiple-statements-on-one-line-colon 	Multiple statements on one line (colon) 	🛠️ 🧪
E702 	multiple-statements-on-one-line-semicolon 	Multiple statements on one line (semicolon) 	🛠️ 🧪
E703 	useless-semicolon 	Statement ends with an unnecessary semicolon 	🛠️ 🧪
E711 	none-comparison 	Comparison to None should be cond is None 	🛠️ 🧪
E712 	true-false-comparison 	Comparison to True should be cond is True or if cond: 	🛠️ 🧪
E713 	not-in-test 	Test for membership should be not in 	🛠️ 🧪
E714 	not-is-test 	Test for object identity should be is not 	🛠️ 🧪
E721 	type-comparison 	Do not compare types, use isinstance() 	🛠️ 🧪
E722 	bare-except 	Do not use bare except 	🛠️ 🧪
E731 	lambda-assignment 	Do not assign a lambda expression, use a def 	🛠️ 🧪
E741 	ambiguous-variable-name 	Ambiguous variable name: {name} 	🛠️ 🧪
E742 	ambiguous-class-name 	Ambiguous class name: {name} 	🛠️ 🧪
E743 	ambiguous-function-name 	Ambiguous function name: {name} 	🛠️ 🧪
E902 	io-error 	{message} 	🛠️ 🧪
E999 	syntax-error 	SyntaxError: 	🛠️ 🧪
mccabe (C90)#

For more, see mccabe on PyPI.

For related settings, see mccabe.
Code 	Name 	Message 	
C901 	complex-structure 	{name} is too complex ({complexity} > {max_complexity}) 	🛠️ 🧪
isort (I)#

For more, see isort on PyPI.

For related settings, see isort.
Code 	Name 	Message 	
I001 	unsorted-imports 	Import block is un-sorted or un-formatted 	🛠️ 🧪
I002 	missing-required-import 	Missing required import: {name} 	🛠️ 🧪
pep8-naming (N)#

For more, see pep8-naming on PyPI.

For related settings, see pep8-naming.
Code 	Name 	Message 	
N801 	invalid-class-name 	Class name {name} should use CapWords convention 	🛠️ 🧪
N802 	invalid-function-name 	Function name {name} should be lowercase 	🛠️ 🧪
N803 	invalid-argument-name 	Argument name {name} should be lowercase 	🛠️ 🧪
N804 	invalid-first-argument-name-for-class-method 	First argument of a class method should be named cls 	🛠️ 🧪
N805 	invalid-first-argument-name-for-method 	First argument of a method should be named self 	🛠️ 🧪
N806 	non-lowercase-variable-in-function 	Variable {name} in function should be lowercase 	🛠️ 🧪
N807 	dunder-function-name 	Function name should not start and end with __ 	🛠️ 🧪
N811 	constant-imported-as-non-constant 	Constant {name} imported as non-constant {asname} 	🛠️ 🧪
N812 	lowercase-imported-as-non-lowercase 	Lowercase {name} imported as non-lowercase {asname} 	🛠️ 🧪
N813 	camelcase-imported-as-lowercase 	Camelcase {name} imported as lowercase {asname} 	🛠️ 🧪
N814 	camelcase-imported-as-constant 	Camelcase {name} imported as constant {asname} 	🛠️ 🧪
N815 	mixed-case-variable-in-class-scope 	Variable {name} in class scope should not be mixedCase 	🛠️ 🧪
N816 	mixed-case-variable-in-global-scope 	Variable {name} in global scope should not be mixedCase 	🛠️ 🧪
N817 	camelcase-imported-as-acronym 	CamelCase {name} imported as acronym {asname} 	🛠️ 🧪
N818 	error-suffix-on-exception-name 	Exception name {name} should be named with an Error suffix 	🛠️ 🧪
N999 	invalid-module-name 	Invalid module name: '{name}' 	🛠️ 🧪
pydocstyle (D)#

For more, see pydocstyle on PyPI.

For related settings, see pydocstyle.
Code 	Name 	Message 	
D100 	undocumented-public-module 	Missing docstring in public module 	🛠️ 🧪
D101 	undocumented-public-class 	Missing docstring in public class 	🛠️ 🧪
D102 	undocumented-public-method 	Missing docstring in public method 	🛠️ 🧪
D103 	undocumented-public-function 	Missing docstring in public function 	🛠️ 🧪
D104 	undocumented-public-package 	Missing docstring in public package 	🛠️ 🧪
D105 	undocumented-magic-method 	Missing docstring in magic method 	🛠️ 🧪
D106 	undocumented-public-nested-class 	Missing docstring in public nested class 	🛠️ 🧪
D107 	undocumented-public-init 	Missing docstring in __init__ 	🛠️ 🧪
D200 	fits-on-one-line 	One-line docstring should fit on one line 	🛠️ 🧪
D201 	no-blank-line-before-function 	No blank lines allowed before function docstring (found {num_lines}) 	🛠️ 🧪
D202 	no-blank-line-after-function 	No blank lines allowed after function docstring (found {num_lines}) 	🛠️ 🧪
D203 	one-blank-line-before-class 	1 blank line required before class docstring 	🛠️ 🧪
D204 	one-blank-line-after-class 	1 blank line required after class docstring 	🛠️ 🧪
D205 	blank-line-after-summary 	1 blank line required between summary line and description 	🛠️ 🧪
D206 	indent-with-spaces 	Docstring should be indented with spaces, not tabs 	🛠️ 🧪
D207 	under-indentation 	Docstring is under-indented 	🛠️ 🧪
D208 	over-indentation 	Docstring is over-indented 	🛠️ 🧪
D209 	new-line-after-last-paragraph 	Multi-line docstring closing quotes should be on a separate line 	🛠️ 🧪
D210 	surrounding-whitespace 	No whitespaces allowed surrounding docstring text 	🛠️ 🧪
D211 	blank-line-before-class 	No blank lines allowed before class docstring 	🛠️ 🧪
D212 	multi-line-summary-first-line 	Multi-line docstring summary should start at the first line 	🛠️ 🧪
D213 	multi-line-summary-second-line 	Multi-line docstring summary should start at the second line 	🛠️ 🧪
D214 	section-not-over-indented 	Section is over-indented ("{name}") 	🛠️ 🧪
D215 	section-underline-not-over-indented 	Section underline is over-indented ("{name}") 	🛠️ 🧪
D300 	triple-single-quotes 	Use triple double quotes """ 	🛠️ 🧪
D301 	escape-sequence-in-docstring 	Use r""" if any backslashes in a docstring 	🛠️ 🧪
D400 	ends-in-period 	First line should end with a period 	🛠️ 🧪
D401 	non-imperative-mood 	First line of docstring should be in imperative mood: "{first_line}" 	🛠️ 🧪
D402 	no-signature 	First line should not be the function's signature 	🛠️ 🧪
D403 	first-line-capitalized 	First word of the first line should be capitalized: {} -> {} 	🛠️ 🧪
D404 	docstring-starts-with-this 	First word of the docstring should not be "This" 	🛠️ 🧪
D405 	capitalize-section-name 	Section name should be properly capitalized ("{name}") 	🛠️ 🧪
D406 	new-line-after-section-name 	Section name should end with a newline ("{name}") 	🛠️ 🧪
D407 	dashed-underline-after-section 	Missing dashed underline after section ("{name}") 	🛠️ 🧪
D408 	section-underline-after-name 	Section underline should be in the line following the section's name ("{name}") 	🛠️ 🧪
D409 	section-underline-matches-section-length 	Section underline should match the length of its name ("{name}") 	🛠️ 🧪
D410 	no-blank-line-after-section 	Missing blank line after section ("{name}") 	🛠️ 🧪
D411 	no-blank-line-before-section 	Missing blank line before section ("{name}") 	🛠️ 🧪
D412 	blank-lines-between-header-and-content 	No blank lines allowed between a section header and its content ("{name}") 	🛠️ 🧪
D413 	blank-line-after-last-section 	Missing blank line after last section ("{name}") 	🛠️ 🧪
D414 	empty-docstring-section 	Section has no content ("{name}") 	🛠️ 🧪
D415 	ends-in-punctuation 	First line should end with a period, question mark, or exclamation point 	🛠️ 🧪
D416 	section-name-ends-in-colon 	Section name should end with a colon ("{name}") 	🛠️ 🧪
D417 	undocumented-param 	Missing argument description in the docstring for {definition}: {name} 	🛠️ 🧪
D418 	overload-with-docstring 	Function decorated with @overload shouldn't contain a docstring 	🛠️ 🧪
D419 	empty-docstring 	Docstring is empty 	🛠️ 🧪
pyupgrade (UP)#

For more, see pyupgrade on PyPI.

For related settings, see pyupgrade.
Code 	Name 	Message 	
UP001 	useless-metaclass-type 	__metaclass__ = type is implied 	🛠️ 🧪
UP003 	type-of-primitive 	Use {} instead of type(...) 	🛠️ 🧪
UP004 	useless-object-inheritance 	Class {name} inherits from object 	🛠️ 🧪
UP005 	deprecated-unittest-alias 	{alias} is deprecated, use {target} 	🛠️ 🧪
UP006 	non-pep585-annotation 	Use {to} instead of {from} for type annotation 	🛠️ 🧪
UP007 	non-pep604-annotation 	Use X | Y for type annotations 	🛠️ 🧪
UP008 	super-call-with-parameters 	Use super() instead of super(__class__, self) 	🛠️ 🧪
UP009 	utf8-encoding-declaration 	UTF-8 encoding declaration is unnecessary 	🛠️ 🧪
UP010 	unnecessary-future-import 	Unnecessary __future__ import {import} for target Python version 	🛠️ 🧪
UP011 	lru-cache-without-parameters 	Unnecessary parentheses to functools.lru_cache 	🛠️ 🧪
UP012 	unnecessary-encode-utf8 	Unnecessary call to encode as UTF-8 	🛠️ 🧪
UP013 	convert-typed-dict-functional-to-class 	Convert {name} from TypedDict functional to class syntax 	🛠️ 🧪
UP014 	convert-named-tuple-functional-to-class 	Convert {name} from NamedTuple functional to class syntax 	🛠️ 🧪
UP015 	redundant-open-modes 	Unnecessary open mode parameters 	🛠️ 🧪
UP017 	datetime-timezone-utc 	Use datetime.UTC alias 	🛠️ 🧪
UP018 	native-literals 	Unnecessary {literal_type} call (rewrite as a literal) 	🛠️ 🧪
UP019 	typing-text-str-alias 	typing.Text is deprecated, use str 	🛠️ 🧪
UP020 	open-alias 	Use builtin open 	🛠️ 🧪
UP021 	replace-universal-newlines 	universal_newlines is deprecated, use text 	🛠️ 🧪
UP022 	replace-stdout-stderr 	Sending stdout and stderr to PIPE is deprecated, use capture_output 	🛠️ 🧪
UP023 	deprecated-c-element-tree 	cElementTree is deprecated, use ElementTree 	🛠️ 🧪
UP024 	os-error-alias 	Replace aliased errors with OSError 	🛠️ 🧪
UP025 	unicode-kind-prefix 	Remove unicode literals from strings 	🛠️ 🧪
UP026 	deprecated-mock-import 	mock is deprecated, use unittest.mock 	🛠️ 🧪
UP027 	unpacked-list-comprehension 	Replace unpacked list comprehension with a generator expression 	🛠️ 🧪
UP028 	yield-in-for-loop 	Replace yield over for loop with yield from 	🛠️ 🧪
UP029 	unnecessary-builtin-import 	Unnecessary builtin import: {import} 	🛠️ 🧪
UP030 	format-literals 	Use implicit references for positional format fields 	🛠️ 🧪
UP031 	printf-string-formatting 	Use format specifiers instead of percent format 	🛠️ 🧪
UP032 	f-string 	Use f-string instead of format call 	🛠️ 🧪
UP033 	lru-cache-with-maxsize-none 	Use @functools.cache instead of @functools.lru_cache(maxsize=None) 	🛠️ 🧪
UP034 	extraneous-parentheses 	Avoid extraneous parentheses 	🛠️ 🧪
UP035 	deprecated-import 	Import from {target} instead: 	🛠️ 🧪
UP036 	outdated-version-block 	Version block is outdated for minimum Python version 	🛠️ 🧪
UP037 	quoted-annotation 	Remove quotes from type annotation 	🛠️ 🧪
UP038 	non-pep604-isinstance 	Use X | Y in {} call instead of (X, Y) 	🛠️ 🧪
UP039 	unnecessary-class-parentheses 	Unnecessary parentheses after class definition 	🛠️ 🧪
UP040 	non-pep695-type-alias 	Type alias {name} uses TypeAlias annotation instead of the type keyword 	🛠️ 🧪
flake8-2020 (YTT)#

For more, see flake8-2020 on PyPI.
Code 	Name 	Message 	
YTT101 	sys-version-slice3 	sys.version[:3] referenced (python3.10), use sys.version_info 	🛠️ 🧪
YTT102 	sys-version2 	sys.version[2] referenced (python3.10), use sys.version_info 	🛠️ 🧪
YTT103 	sys-version-cmp-str3 	sys.version compared to string (python3.10), use sys.version_info 	🛠️ 🧪
YTT201 	sys-version-info0-eq3 	sys.version_info[0] == 3 referenced (python4), use >= 	🛠️ 🧪
YTT202 	six-py3 	six.PY3 referenced (python4), use not six.PY2 	🛠️ 🧪
YTT203 	sys-version-info1-cmp-int 	sys.version_info[1] compared to integer (python4), compare sys.version_info to tuple 	🛠️ 🧪
YTT204 	sys-version-info-minor-cmp-int 	sys.version_info.minor compared to integer (python4), compare sys.version_info to tuple 	🛠️ 🧪
YTT301 	sys-version0 	sys.version[0] referenced (python10), use sys.version_info 	🛠️ 🧪
YTT302 	sys-version-cmp-str10 	sys.version compared to string (python10), use sys.version_info 	🛠️ 🧪
YTT303 	sys-version-slice1 	sys.version[:1] referenced (python10), use sys.version_info 	🛠️ 🧪
flake8-annotations (ANN)#

For more, see flake8-annotations on PyPI.

For related settings, see flake8-annotations.
Code 	Name 	Message 	
ANN001 	missing-type-function-argument 	Missing type annotation for function argument {name} 	🛠️ 🧪
ANN002 	missing-type-args 	Missing type annotation for *{name} 	🛠️ 🧪
ANN003 	missing-type-kwargs 	Missing type annotation for **{name} 	🛠️ 🧪
ANN101 	missing-type-self 	Missing type annotation for {name} in method 	🛠️ 🧪
ANN102 	missing-type-cls 	Missing type annotation for {name} in classmethod 	🛠️ 🧪
ANN201 	missing-return-type-undocumented-public-function 	Missing return type annotation for public function {name} 	🛠️ 🧪
ANN202 	missing-return-type-private-function 	Missing return type annotation for private function {name} 	🛠️ 🧪
ANN204 	missing-return-type-special-method 	Missing return type annotation for special method {name} 	🛠️ 🧪
ANN205 	missing-return-type-static-method 	Missing return type annotation for staticmethod {name} 	🛠️ 🧪
ANN206 	missing-return-type-class-method 	Missing return type annotation for classmethod {name} 	🛠️ 🧪
ANN401 	any-type 	Dynamically typed expressions (typing.Any) are disallowed in {name} 	🛠️ 🧪
flake8-async (ASYNC)#

For more, see flake8-async on PyPI.
Code 	Name 	Message 	
ASYNC100 	blocking-http-call-in-async-function 	Async functions should not call blocking HTTP methods 	🛠️ 🧪
ASYNC101 	open-sleep-or-subprocess-in-async-function 	Async functions should not call open, time.sleep, or subprocess methods 	🛠️ 🧪
ASYNC102 	blocking-os-call-in-async-function 	Async functions should not call synchronous os methods 	🛠️ 🧪
flake8-bandit (S)#

For more, see flake8-bandit on PyPI.

For related settings, see flake8-bandit.
Code 	Name 	Message 	
S101 	assert 	Use of assert detected 	🛠️ 🧪
S102 	exec-builtin 	Use of exec detected 	🛠️ 🧪
S103 	bad-file-permissions 	os.chmod setting a permissive mask {mask:#o} on file or directory 	🛠️ 🧪
S104 	hardcoded-bind-all-interfaces 	Possible binding to all interfaces 	🛠️ 🧪
S105 	hardcoded-password-string 	Possible hardcoded password assigned to: "{}" 	🛠️ 🧪
S106 	hardcoded-password-func-arg 	Possible hardcoded password assigned to argument: "{}" 	🛠️ 🧪
S107 	hardcoded-password-default 	Possible hardcoded password assigned to function default: "{}" 	🛠️ 🧪
S108 	hardcoded-temp-file 	Probable insecure usage of temporary file or directory: "{}" 	🛠️ 🧪
S110 	try-except-pass 	try-except-pass detected, consider logging the exception 	🛠️ 🧪
S112 	try-except-continue 	try-except-continue detected, consider logging the exception 	🛠️ 🧪
S113 	request-without-timeout 	Probable use of requests call without timeout 	🛠️ 🧪
S201 	flask-debug-true 	Use of debug=True in Flask app detected 	🛠️ 🧪
S301 	suspicious-pickle-usage 	pickle and modules that wrap it can be unsafe when used to deserialize untrusted data, possible security issue 	🛠️ 🧪
S302 	suspicious-marshal-usage 	Deserialization with the marshal module is possibly dangerous 	🛠️ 🧪
S303 	suspicious-insecure-hash-usage 	Use of insecure MD2, MD4, MD5, or SHA1 hash function 	🛠️ 🧪
S304 	suspicious-insecure-cipher-usage 	Use of insecure cipher, replace with a known secure cipher such as AES 	🛠️ 🧪
S305 	suspicious-insecure-cipher-mode-usage 	Use of insecure cipher mode, replace with a known secure cipher such as AES 	🛠️ 🧪
S306 	suspicious-mktemp-usage 	Use of insecure and deprecated function (mktemp) 	🛠️ 🧪
S307 	suspicious-eval-usage 	Use of possibly insecure function; consider using ast.literal_eval 	🛠️ 🧪
S308 	suspicious-mark-safe-usage 	Use of mark_safe may expose cross-site scripting vulnerabilities 	🛠️ 🧪
S310 	suspicious-url-open-usage 	Audit URL open for permitted schemes. Allowing use of file: or custom schemes is often unexpected. 	🛠️ 🧪
S311 	suspicious-non-cryptographic-random-usage 	Standard pseudo-random generators are not suitable for cryptographic purposes 	🛠️ 🧪
S312 	suspicious-telnet-usage 	Telnet-related functions are being called. Telnet is considered insecure. Use SSH or some other encrypted protocol. 	🛠️ 🧪
S313 	suspicious-xmlc-element-tree-usage 	Using xml to parse untrusted data is known to be vulnerable to XML attacks; use defusedxml equivalents 	🛠️ 🧪
S314 	suspicious-xml-element-tree-usage 	Using xml to parse untrusted data is known to be vulnerable to XML attacks; use defusedxml equivalents 	🛠️ 🧪
S315 	suspicious-xml-expat-reader-usage 	Using xml to parse untrusted data is known to be vulnerable to XML attacks; use defusedxml equivalents 	🛠️ 🧪
S316 	suspicious-xml-expat-builder-usage 	Using xml to parse untrusted data is known to be vulnerable to XML attacks; use defusedxml equivalents 	🛠️ 🧪
S317 	suspicious-xml-sax-usage 	Using xml to parse untrusted data is known to be vulnerable to XML attacks; use defusedxml equivalents 	🛠️ 🧪
S318 	suspicious-xml-mini-dom-usage 	Using xml to parse untrusted data is known to be vulnerable to XML attacks; use defusedxml equivalents 	🛠️ 🧪
S319 	suspicious-xml-pull-dom-usage 	Using xml to parse untrusted data is known to be vulnerable to XML attacks; use defusedxml equivalents 	🛠️ 🧪
S320 	suspicious-xmle-tree-usage 	Using lxml to parse untrusted data is known to be vulnerable to XML attacks 	🛠️ 🧪
S321 	suspicious-ftp-lib-usage 	FTP-related functions are being called. FTP is considered insecure. Use SSH/SFTP/SCP or some other encrypted protocol. 	🛠️ 🧪
S323 	suspicious-unverified-context-usage 	Python allows using an insecure context via the _create_unverified_context that reverts to the previous behavior that does not validate certificates or perform hostname checks. 	🛠️ 🧪
S324 	hashlib-insecure-hash-function 	Probable use of insecure hash functions in hashlib: {string} 	🛠️ 🧪
S501 	request-with-no-cert-validation 	Probable use of {string} call with verify=False disabling SSL certificate checks 	🛠️ 🧪
S506 	unsafe-yaml-load 	Probable use of unsafe loader {name} with yaml.load. Allows instantiation of arbitrary objects. Consider yaml.safe_load. 	🛠️ 🧪
S507 	ssh-no-host-key-verification 	Paramiko call with policy set to automatically trust the unknown host key 	🛠️ 🧪
S508 	snmp-insecure-version 	The use of SNMPv1 and SNMPv2 is insecure. Use SNMPv3 if able. 	🛠️ 🧪
S509 	snmp-weak-cryptography 	You should not use SNMPv3 without encryption. noAuthNoPriv & authNoPriv is insecure. 	🛠️ 🧪
S601 	paramiko-call 	Possible shell injection via Paramiko call; check inputs are properly sanitized 	🛠️ 🧪
S602 	subprocess-popen-with-shell-equals-true 	subprocess call with shell=True seems safe, but may be changed in the future; consider rewriting without shell 	🛠️ 🧪
S603 	subprocess-without-shell-equals-true 	subprocess call: check for execution of untrusted input 	🛠️ 🧪
S604 	call-with-shell-equals-true 	Function call with shell=True parameter identified, security issue 	🛠️ 🧪
S605 	start-process-with-a-shell 	Starting a process with a shell: seems safe, but may be changed in the future; consider rewriting without shell 	🛠️ 🧪
S606 	start-process-with-no-shell 	Starting a process without a shell 	🛠️ 🧪
S607 	start-process-with-partial-path 	Starting a process with a partial executable path 	🛠️ 🧪
S608 	hardcoded-sql-expression 	Possible SQL injection vector through string-based query construction 	🛠️ 🧪
S609 	unix-command-wildcard-injection 	Possible wildcard injection in call due to * usage 	🛠️ 🧪
S612 	logging-config-insecure-listen 	Use of insecure logging.config.listen detected 	🛠️ 🧪
S701 	jinja2-autoescape-false 	Using jinja2 templates with autoescape=False is dangerous and can lead to XSS. Ensure autoescape=True or use the select_autoescape function. 	🛠️ 🧪
flake8-blind-except (BLE)#

For more, see flake8-blind-except on PyPI.
Code 	Name 	Message 	
BLE001 	blind-except 	Do not catch blind exception: {name} 	🛠️ 🧪
flake8-boolean-trap (FBT)#

For more, see flake8-boolean-trap on PyPI.
Code 	Name 	Message 	
FBT001 	boolean-type-hint-positional-argument 	Boolean-typed positional argument in function definition 	🛠️ 🧪
FBT002 	boolean-default-value-positional-argument 	Boolean default positional argument in function definition 	🛠️ 🧪
FBT003 	boolean-positional-value-in-call 	Boolean positional value in function call 	🛠️ 🧪
flake8-bugbear (B)#

For more, see flake8-bugbear on PyPI.

For related settings, see flake8-bugbear.
Code 	Name 	Message 	
B002 	unary-prefix-increment-decrement 	Python does not support the unary prefix increment operator (++) 	🛠️ 🧪
B003 	assignment-to-os-environ 	Assigning to os.environ doesn't clear the environment 	🛠️ 🧪
B004 	unreliable-callable-check 	Using hasattr(x, "__call__") to test if x is callable is unreliable. Use callable(x) for consistent results. 	🛠️ 🧪
B005 	strip-with-multi-characters 	Using .strip() with multi-character strings is misleading 	🛠️ 🧪
B006 	mutable-argument-default 	Do not use mutable data structures for argument defaults 	🛠️ 🧪
B007 	unused-loop-control-variable 	Loop control variable {name} not used within loop body 	🛠️ 🧪
B008 	function-call-in-default-argument 	Do not perform function call {name} in argument defaults 	🛠️ 🧪
B009 	get-attr-with-constant 	Do not call getattr with a constant attribute value. It is not any safer than normal property access. 	🛠️ 🧪
B010 	set-attr-with-constant 	Do not call setattr with a constant attribute value. It is not any safer than normal property access. 	🛠️ 🧪
B011 	assert-false 	Do not assert False (python -O removes these calls), raise AssertionError() 	🛠️ 🧪
B012 	jump-statement-in-finally 	{name} inside finally blocks cause exceptions to be silenced 	🛠️ 🧪
B013 	redundant-tuple-in-exception-handler 	A length-one tuple literal is redundant in exception handlers 	🛠️ 🧪
B014 	duplicate-handler-exception 	Exception handler with duplicate exception: {name} 	🛠️ 🧪
B015 	useless-comparison 	Pointless comparison. This comparison does nothing but waste CPU instructions. Either prepend assert or remove it. 	🛠️ 🧪
B016 	raise-literal 	Cannot raise a literal. Did you intend to return it or raise an Exception? 	🛠️ 🧪
B017 	assert-raises-exception 	{assertion}({exception}) should be considered evil 	🛠️ 🧪
B018 	useless-expression 	Found useless expression. Either assign it to a variable or remove it. 	🛠️ 🧪
B019 	cached-instance-method 	Use of functools.lru_cache or functools.cache on methods can lead to memory leaks 	🛠️ 🧪
B020 	loop-variable-overrides-iterator 	Loop control variable {name} overrides iterable it iterates 	🛠️ 🧪
B021 	f-string-docstring 	f-string used as docstring. Python will interpret this as a joined string, rather than a docstring. 	🛠️ 🧪
B022 	useless-contextlib-suppress 	No arguments passed to contextlib.suppress. No exceptions will be suppressed and therefore this context manager is redundant 	🛠️ 🧪
B023 	function-uses-loop-variable 	Function definition does not bind loop variable {name} 	🛠️ 🧪
B024 	abstract-base-class-without-abstract-method 	{name} is an abstract base class, but it has no abstract methods 	🛠️ 🧪
B025 	duplicate-try-block-exception 	try-except block with duplicate exception {name} 	🛠️ 🧪
B026 	star-arg-unpacking-after-keyword-arg 	Star-arg unpacking after a keyword argument is strongly discouraged 	🛠️ 🧪
B027 	empty-method-without-abstract-decorator 	{name} is an empty method in an abstract base class, but has no abstract decorator 	🛠️ 🧪
B028 	no-explicit-stacklevel 	No explicit stacklevel keyword argument found 	🛠️ 🧪
B029 	except-with-empty-tuple 	Using except (): with an empty tuple does not catch anything; add exceptions to handle 	🛠️ 🧪
B030 	except-with-non-exception-classes 	except handlers should only be exception classes or tuples of exception classes 	🛠️ 🧪
B031 	reuse-of-groupby-generator 	Using the generator returned from itertools.groupby() more than once will do nothing on the second usage 	🛠️ 🧪
B032 	unintentional-type-annotation 	Possible unintentional type annotation (using :). Did you mean to assign (using =)? 	🛠️ 🧪
B033 	duplicate-value 	Sets should not contain duplicate item {value} 	🛠️ 🧪
B034 	re-sub-positional-args 	{method} should pass {param_name} and flags as keyword arguments to avoid confusion due to unintuitive argument positions 	🛠️ 🧪
B904 	raise-without-from-inside-except 	Within an except clause, raise exceptions with raise ... from err or raise ... from None to distinguish them from errors in exception handling 	🛠️ 🧪
B905 	zip-without-explicit-strict 	zip() without an explicit strict= parameter 	🛠️ 🧪
flake8-builtins (A)#

For more, see flake8-builtins on PyPI.

For related settings, see flake8-builtins.
Code 	Name 	Message 	
A001 	builtin-variable-shadowing 	Variable {name} is shadowing a Python builtin 	🛠️ 🧪
A002 	builtin-argument-shadowing 	Argument {name} is shadowing a Python builtin 	🛠️ 🧪
A003 	builtin-attribute-shadowing 	Class attribute {name} is shadowing a Python builtin 	🛠️ 🧪
flake8-commas (COM)#

For more, see flake8-commas on PyPI.
Code 	Name 	Message 	
COM812 	missing-trailing-comma 	Trailing comma missing 	🛠️ 🧪
COM818 	trailing-comma-on-bare-tuple 	Trailing comma on bare tuple prohibited 	🛠️ 🧪
COM819 	prohibited-trailing-comma 	Trailing comma prohibited 	🛠️ 🧪
flake8-copyright (CPY)#

For more, see flake8-copyright on PyPI.

For related settings, see flake8-copyright.
Code 	Name 	Message 	
CPY001 	missing-copyright-notice 	Missing copyright notice at top of file 	🛠️ 🧪
flake8-comprehensions (C4)#

For more, see flake8-comprehensions on PyPI.

For related settings, see flake8-comprehensions.
Code 	Name 	Message 	
C400 	unnecessary-generator-list 	Unnecessary generator (rewrite as a list comprehension) 	🛠️ 🧪
C401 	unnecessary-generator-set 	Unnecessary generator (rewrite as a set comprehension) 	🛠️ 🧪
C402 	unnecessary-generator-dict 	Unnecessary generator (rewrite as a dict comprehension) 	🛠️ 🧪
C403 	unnecessary-list-comprehension-set 	Unnecessary list comprehension (rewrite as a set comprehension) 	🛠️ 🧪
C404 	unnecessary-list-comprehension-dict 	Unnecessary list comprehension (rewrite as a dict comprehension) 	🛠️ 🧪
C405 	unnecessary-literal-set 	Unnecessary {obj_type} literal (rewrite as a set literal) 	🛠️ 🧪
C406 	unnecessary-literal-dict 	Unnecessary {obj_type} literal (rewrite as a dict literal) 	🛠️ 🧪
C408 	unnecessary-collection-call 	Unnecessary {obj_type} call (rewrite as a literal) 	🛠️ 🧪
C409 	unnecessary-literal-within-tuple-call 	Unnecessary {literal} literal passed to tuple() (rewrite as a tuple literal) 	🛠️ 🧪
C410 	unnecessary-literal-within-list-call 	Unnecessary {literal} literal passed to list() (remove the outer call to list()) 	🛠️ 🧪
C411 	unnecessary-list-call 	Unnecessary list call (remove the outer call to list()) 	🛠️ 🧪
C413 	unnecessary-call-around-sorted 	Unnecessary {func} call around sorted() 	🛠️ 🧪
C414 	unnecessary-double-cast-or-process 	Unnecessary {inner} call within {outer}() 	🛠️ 🧪
C415 	unnecessary-subscript-reversal 	Unnecessary subscript reversal of iterable within {func}() 	🛠️ 🧪
C416 	unnecessary-comprehension 	Unnecessary {obj_type} comprehension (rewrite using {obj_type}()) 	🛠️ 🧪
C417 	unnecessary-map 	Unnecessary map usage (rewrite using a {object_type}) 	🛠️ 🧪
C418 	unnecessary-literal-within-dict-call 	Unnecessary dict {kind} passed to dict() (remove the outer call to dict()) 	🛠️ 🧪
C419 	unnecessary-comprehension-any-all 	Unnecessary list comprehension. 	🛠️ 🧪
flake8-datetimez (DTZ)#

For more, see flake8-datetimez on PyPI.
Code 	Name 	Message 	
DTZ001 	call-datetime-without-tzinfo 	The use of datetime.datetime() without tzinfo argument is not allowed 	🛠️ 🧪
DTZ002 	call-datetime-today 	The use of datetime.datetime.today() is not allowed, use datetime.datetime.now(tz=) instead 	🛠️ 🧪
DTZ003 	call-datetime-utcnow 	The use of datetime.datetime.utcnow() is not allowed, use datetime.datetime.now(tz=) instead 	🛠️ 🧪
DTZ004 	call-datetime-utcfromtimestamp 	The use of datetime.datetime.utcfromtimestamp() is not allowed, use datetime.datetime.fromtimestamp(ts, tz=) instead 	🛠️ 🧪
DTZ005 	call-datetime-now-without-tzinfo 	The use of datetime.datetime.now() without tz argument is not allowed 	🛠️ 🧪
DTZ006 	call-datetime-fromtimestamp 	The use of datetime.datetime.fromtimestamp() without tz argument is not allowed 	🛠️ 🧪
DTZ007 	call-datetime-strptime-without-zone 	The use of datetime.datetime.strptime() without %z must be followed by .replace(tzinfo=) or .astimezone() 	🛠️ 🧪
DTZ011 	call-date-today 	The use of datetime.date.today() is not allowed, use datetime.datetime.now(tz=).date() instead 	🛠️ 🧪
DTZ012 	call-date-fromtimestamp 	The use of datetime.date.fromtimestamp() is not allowed, use datetime.datetime.fromtimestamp(ts, tz=).date() instead 	🛠️ 🧪
flake8-debugger (T10)#

For more, see flake8-debugger on PyPI.
Code 	Name 	Message 	
T100 	debugger 	Trace found: {name} used 	🛠️ 🧪
flake8-django (DJ)#

For more, see flake8-django on PyPI.
Code 	Name 	Message 	
DJ001 	django-nullable-model-string-field 	Avoid using null=True on string-based fields such as 	🛠️ 🧪
DJ003 	django-locals-in-render-function 	Avoid passing locals() as context to a render function 	🛠️ 🧪
DJ006 	django-exclude-with-model-form 	Do not use exclude with ModelForm, use fields instead 	🛠️ 🧪
DJ007 	django-all-with-model-form 	Do not use __all__ with ModelForm, use fields instead 	🛠️ 🧪
DJ008 	django-model-without-dunder-str 	Model does not define __str__ method 	🛠️ 🧪
DJ012 	django-unordered-body-content-in-model 	Order of model's inner classes, methods, and fields does not follow the Django Style Guide: {element_type} should come before 	🛠️ 🧪
DJ013 	django-non-leading-receiver-decorator 	@receiver decorator must be on top of all the other decorators 	🛠️ 🧪
flake8-errmsg (EM)#

For more, see flake8-errmsg on PyPI.

For related settings, see flake8-errmsg.
Code 	Name 	Message 	
EM101 	raw-string-in-exception 	Exception must not use a string literal, assign to variable first 	🛠️ 🧪
EM102 	f-string-in-exception 	Exception must not use an f-string literal, assign to variable first 	🛠️ 🧪
EM103 	dot-format-in-exception 	Exception must not use a .format() string directly, assign to variable first 	🛠️ 🧪
flake8-executable (EXE)#

For more, see flake8-executable on PyPI.
Code 	Name 	Message 	
EXE001 	shebang-not-executable 	Shebang is present but file is not executable 	🛠️ 🧪
EXE002 	shebang-missing-executable-file 	The file is executable but no shebang is present 	🛠️ 🧪
EXE003 	shebang-missing-python 	Shebang should contain python 	🛠️ 🧪
EXE004 	shebang-leading-whitespace 	Avoid whitespace before shebang 	🛠️ 🧪
EXE005 	shebang-not-first-line 	Shebang should be at the beginning of the file 	🛠️ 🧪
flake8-future-annotations (FA)#

For more, see flake8-future-annotations on PyPI.
Code 	Name 	Message 	
FA100 	future-rewritable-type-annotation 	Missing from __future__ import annotations, but uses {name} 	🛠️ 🧪
FA102 	future-required-type-annotation 	Missing from __future__ import annotations, but uses 	🛠️ 🧪
flake8-implicit-str-concat (ISC)#

For more, see flake8-implicit-str-concat on PyPI.

For related settings, see flake8-implicit-str-concat.
Code 	Name 	Message 	
ISC001 	single-line-implicit-string-concatenation 	Implicitly concatenated string literals on one line 	🛠️ 🧪
ISC002 	multi-line-implicit-string-concatenation 	Implicitly concatenated string literals over multiple lines 	🛠️ 🧪
ISC003 	explicit-string-concatenation 	Explicitly concatenated string should be implicitly concatenated 	🛠️ 🧪
flake8-import-conventions (ICN)#

For more, see flake8-import-conventions on GitHub.

For related settings, see flake8-import-conventions.
Code 	Name 	Message 	
ICN001 	unconventional-import-alias 	{name} should be imported as {asname} 	🛠️ 🧪
ICN002 	banned-import-alias 	{name} should not be imported as {asname} 	🛠️ 🧪
ICN003 	banned-import-from 	Members of {name} should not be imported explicitly 	🛠️ 🧪
flake8-logging-format (G)#

For more, see flake8-logging-format on PyPI.
Code 	Name 	Message 	
G001 	logging-string-format 	Logging statement uses str.format 	🛠️ 🧪
G002 	logging-percent-format 	Logging statement uses % 	🛠️ 🧪
G003 	logging-string-concat 	Logging statement uses + 	🛠️ 🧪
G004 	logging-f-string 	Logging statement uses f-string 	🛠️ 🧪
G010 	logging-warn 	Logging statement uses warn instead of warning 	🛠️ 🧪
G101 	logging-extra-attr-clash 	Logging statement uses an extra field that clashes with a LogRecord field: {key} 	🛠️ 🧪
G201 	logging-exc-info 	Logging .exception(...) should be used instead of .error(..., exc_info=True) 	🛠️ 🧪
G202 	logging-redundant-exc-info 	Logging statement has redundant exc_info 	🛠️ 🧪
flake8-no-pep420 (INP)#

For more, see flake8-no-pep420 on PyPI.
Code 	Name 	Message 	
INP001 	implicit-namespace-package 	File {filename} is part of an implicit namespace package. Add an __init__.py. 	🛠️ 🧪
flake8-pie (PIE)#

For more, see flake8-pie on PyPI.
Code 	Name 	Message 	
PIE790 	unnecessary-pass 	Unnecessary pass statement 	🛠️ 🧪
PIE794 	duplicate-class-field-definition 	Class field {name} is defined multiple times 	🛠️ 🧪
PIE796 	non-unique-enums 	Enum contains duplicate value: {value} 	🛠️ 🧪
PIE800 	unnecessary-spread 	Unnecessary spread ** 	🛠️ 🧪
PIE804 	unnecessary-dict-kwargs 	Unnecessary dict kwargs 	🛠️ 🧪
PIE807 	reimplemented-list-builtin 	Prefer list over useless lambda 	🛠️ 🧪
PIE808 	unnecessary-range-start 	Unnecessary start argument in range 	🛠️ 🧪
PIE810 	multiple-starts-ends-with 	Call {attr} once with a tuple 	🛠️ 🧪
flake8-print (T20)#

For more, see flake8-print on PyPI.
Code 	Name 	Message 	
T201 	print 	print found 	🛠️ 🧪
T203 	p-print 	pprint found 	🛠️ 🧪
flake8-pyi (PYI)#

For more, see flake8-pyi on PyPI.
Code 	Name 	Message 	
PYI001 	unprefixed-type-param 	Name of private {kind} must start with _ 	🛠️ 🧪
PYI002 	complex-if-statement-in-stub 	if test must be a simple comparison against sys.platform or sys.version_info 	🛠️ 🧪
PYI003 	unrecognized-version-info-check 	Unrecognized sys.version_info check 	🛠️ 🧪
PYI004 	patch-version-comparison 	Version comparison must use only major and minor version 	🛠️ 🧪
PYI005 	wrong-tuple-length-version-comparison 	Version comparison must be against a length-{expected_length} tuple 	🛠️ 🧪
PYI006 	bad-version-info-comparison 	Use < or >= for sys.version_info comparisons 	🛠️ 🧪
PYI007 	unrecognized-platform-check 	Unrecognized sys.platform check 	🛠️ 🧪
PYI008 	unrecognized-platform-name 	Unrecognized platform {platform} 	🛠️ 🧪
PYI009 	pass-statement-stub-body 	Empty body should contain ..., not pass 	🛠️ 🧪
PYI010 	non-empty-stub-body 	Function body must contain only ... 	🛠️ 🧪
PYI011 	typed-argument-default-in-stub 	Only simple default values allowed for typed arguments 	🛠️ 🧪
PYI012 	pass-in-class-body 	Class body must not contain pass 	🛠️ 🧪
PYI013 	ellipsis-in-non-empty-class-body 	Non-empty class body must not contain ... 	🛠️ 🧪
PYI014 	argument-default-in-stub 	Only simple default values allowed for arguments 	🛠️ 🧪
PYI015 	assignment-default-in-stub 	Only simple default values allowed for assignments 	🛠️ 🧪
PYI016 	duplicate-union-member 	Duplicate union member {} 	🛠️ 🧪
PYI017 	complex-assignment-in-stub 	Stubs should not contain assignments to attributes or multiple targets 	🛠️ 🧪
PYI018 	unused-private-type-var 	Private TypeVar {name} is never used 	🛠️ 🧪
PYI019 	custom-type-var-return-type 	Methods like {method_name} should return typing.Self instead of a custom TypeVar 	🛠️ 🧪
PYI020 	quoted-annotation-in-stub 	Quoted annotations should not be included in stubs 	🛠️ 🧪
PYI021 	docstring-in-stub 	Docstrings should not be included in stubs 	🛠️ 🧪
PYI024 	collections-named-tuple 	Use typing.NamedTuple instead of collections.namedtuple 	🛠️ 🧪
PYI025 	unaliased-collections-abc-set-import 	Use from collections.abc import Set as AbstractSet to avoid confusion with the set builtin 	🛠️ 🧪
PYI026 	type-alias-without-annotation 	Use {module}.TypeAlias for type alias, e.g., {name}: TypeAlias = {value} 	🛠️ 🧪
PYI029 	str-or-repr-defined-in-stub 	Defining {name} in a stub is almost always redundant 	🛠️ 🧪
PYI030 	unnecessary-literal-union 	Multiple literal members in a union. Use a single literal, e.g. Literal[{}] 	🛠️ 🧪
PYI032 	any-eq-ne-annotation 	Prefer object to Any for the second parameter to {method_name} 	🛠️ 🧪
PYI033 	type-comment-in-stub 	Don't use type comments in stub file 	🛠️ 🧪
PYI034 	non-self-return-type 	__new__ methods usually return self at runtime 	🛠️ 🧪
PYI035 	unassigned-special-variable-in-stub 	{name} in a stub file must have a value, as it has the same semantics as {name} at runtime 	🛠️ 🧪
PYI036 	bad-exit-annotation 	Star-args in {method_name} should be annotated with object 	🛠️ 🧪
PYI041 	redundant-numeric-union 	Use {supertype} instead of {subtype} | {supertype} 	🛠️ 🧪
PYI042 	snake-case-type-alias 	Type alias {name} should be CamelCase 	🛠️ 🧪
PYI043 	t-suffixed-type-alias 	Private type alias {name} should not be suffixed with T (the T suffix implies that an object is a TypeVar) 	🛠️ 🧪
PYI044 	future-annotations-in-stub 	from __future__ import annotations has no effect in stub files, since type checkers automatically treat stubs as having those semantics 	🛠️ 🧪
PYI045 	iter-method-return-iterable 	__aiter__ methods should return an AsyncIterator, not an AsyncIterable 	🛠️ 🧪
PYI046 	unused-private-protocol 	Private protocol {name} is never used 	🛠️ 🧪
PYI047 	unused-private-type-alias 	Private TypeAlias {name} is never used 	🛠️ 🧪
PYI048 	stub-body-multiple-statements 	Function body must contain exactly one statement 	🛠️ 🧪
PYI049 	unused-private-typed-dict 	Private TypedDict {name} is never used 	🛠️ 🧪
PYI050 	no-return-argument-annotation-in-stub 	Prefer {module}.Never over NoReturn for argument annotations 	🛠️ 🧪
PYI051 	redundant-literal-union 	Literal[{literal}] is redundant in a union with {builtin_type} 	🛠️ 🧪
PYI052 	unannotated-assignment-in-stub 	Need type annotation for {name} 	🛠️ 🧪
PYI053 	string-or-bytes-too-long 	String and bytes literals longer than 50 characters are not permitted 	🛠️ 🧪
PYI054 	numeric-literal-too-long 	Numeric literals with a string representation longer than ten characters are not permitted 	🛠️ 🧪
PYI055 	unnecessary-type-union 	Multiple type members in a union. Combine them into one, e.g., type[{union_str}]. 	🛠️ 🧪
PYI056 	unsupported-method-call-on-all 	Calling .{name}() on __all__ may not be supported by all type checkers (use += instead) 	🛠️ 🧪
flake8-pytest-style (PT)#

For more, see flake8-pytest-style on PyPI.

For related settings, see flake8-pytest-style.
Code 	Name 	Message 	
PT001 	pytest-fixture-incorrect-parentheses-style 	Use @pytest.fixture{expected} over @pytest.fixture{actual} 	🛠️ 🧪
PT002 	pytest-fixture-positional-args 	Configuration for fixture {function} specified via positional args, use kwargs 	🛠️ 🧪
PT003 	pytest-extraneous-scope-function 	scope='function' is implied in @pytest.fixture() 	🛠️ 🧪
PT004 	pytest-missing-fixture-name-underscore 	Fixture {function} does not return anything, add leading underscore 	🛠️ 🧪
PT005 	pytest-incorrect-fixture-name-underscore 	Fixture {function} returns a value, remove leading underscore 	🛠️ 🧪
PT006 	pytest-parametrize-names-wrong-type 	Wrong name(s) type in @pytest.mark.parametrize, expected {expected} 	🛠️ 🧪
PT007 	pytest-parametrize-values-wrong-type 	Wrong values type in @pytest.mark.parametrize expected {values} of {row} 	🛠️ 🧪
PT008 	pytest-patch-with-lambda 	Use return_value= instead of patching with lambda 	🛠️ 🧪
PT009 	pytest-unittest-assertion 	Use a regular assert instead of unittest-style {assertion} 	🛠️ 🧪
PT010 	pytest-raises-without-exception 	set the expected exception in pytest.raises() 	🛠️ 🧪
PT011 	pytest-raises-too-broad 	pytest.raises({exception}) is too broad, set the match parameter or use a more specific exception 	🛠️ 🧪
PT012 	pytest-raises-with-multiple-statements 	pytest.raises() block should contain a single simple statement 	🛠️ 🧪
PT013 	pytest-incorrect-pytest-import 	Found incorrect import of pytest, use simple import pytest instead 	🛠️ 🧪
PT014 	pytest-duplicate-parametrize-test-cases 	Duplicate of test case at index {index} in @pytest_mark.parametrize 	🛠️ 🧪
PT015 	pytest-assert-always-false 	Assertion always fails, replace with pytest.fail() 	🛠️ 🧪
PT016 	pytest-fail-without-message 	No message passed to pytest.fail() 	🛠️ 🧪
PT017 	pytest-assert-in-except 	Found assertion on exception {name} in except block, use pytest.raises() instead 	🛠️ 🧪
PT018 	pytest-composite-assertion 	Assertion should be broken down into multiple parts 	🛠️ 🧪
PT019 	pytest-fixture-param-without-value 	Fixture {name} without value is injected as parameter, use @pytest.mark.usefixtures instead 	🛠️ 🧪
PT020 	pytest-deprecated-yield-fixture 	@pytest.yield_fixture is deprecated, use @pytest.fixture 	🛠️ 🧪
PT021 	pytest-fixture-finalizer-callback 	Use yield instead of request.addfinalizer 	🛠️ 🧪
PT022 	pytest-useless-yield-fixture 	No teardown in fixture {name}, use return instead of yield 	🛠️ 🧪
PT023 	pytest-incorrect-mark-parentheses-style 	Use @pytest.mark.{mark_name}{expected_parens} over @pytest.mark.{mark_name}{actual_parens} 	🛠️ 🧪
PT024 	pytest-unnecessary-asyncio-mark-on-fixture 	pytest.mark.asyncio is unnecessary for fixtures 	🛠️ 🧪
PT025 	pytest-erroneous-use-fixtures-on-fixture 	pytest.mark.usefixtures has no effect on fixtures 	🛠️ 🧪
PT026 	pytest-use-fixtures-without-parameters 	Useless pytest.mark.usefixtures without parameters 	🛠️ 🧪
PT027 	pytest-unittest-raises-assertion 	Use pytest.raises instead of unittest-style {assertion} 	🛠️ 🧪
flake8-quotes (Q)#

For more, see flake8-quotes on PyPI.

For related settings, see flake8-quotes.
Code 	Name 	Message 	
Q000 	bad-quotes-inline-string 	Single quotes found but double quotes preferred 	🛠️ 🧪
Q001 	bad-quotes-multiline-string 	Single quote multiline found but double quotes preferred 	🛠️ 🧪
Q002 	bad-quotes-docstring 	Single quote docstring found but double quotes preferred 	🛠️ 🧪
Q003 	avoidable-escaped-quote 	Change outer quotes to avoid escaping inner quotes 	🛠️ 🧪
flake8-raise (RSE)#

For more, see flake8-raise on PyPI.
Code 	Name 	Message 	
RSE102 	unnecessary-paren-on-raise-exception 	Unnecessary parentheses on raised exception 	🛠️ 🧪
flake8-return (RET)#

For more, see flake8-return on PyPI.
Code 	Name 	Message 	
RET501 	unnecessary-return-none 	Do not explicitly return None in function if it is the only possible return value 	🛠️ 🧪
RET502 	implicit-return-value 	Do not implicitly return None in function able to return non-None value 	🛠️ 🧪
RET503 	implicit-return 	Missing explicit return at the end of function able to return non-None value 	🛠️ 🧪
RET504 	unnecessary-assign 	Unnecessary assignment to {name} before return statement 	🛠️ 🧪
RET505 	superfluous-else-return 	Unnecessary {branch} after return statement 	🛠️ 🧪
RET506 	superfluous-else-raise 	Unnecessary {branch} after raise statement 	🛠️ 🧪
RET507 	superfluous-else-continue 	Unnecessary {branch} after continue statement 	🛠️ 🧪
RET508 	superfluous-else-break 	Unnecessary {branch} after break statement 	🛠️ 🧪
flake8-self (SLF)#

For more, see flake8-self on PyPI.

For related settings, see flake8-self.
Code 	Name 	Message 	
SLF001 	private-member-access 	Private member accessed: {access} 	🛠️ 🧪
flake8-slots (SLOT)#

For more, see flake8-slots on PyPI.
Code 	Name 	Message 	
SLOT000 	no-slots-in-str-subclass 	Subclasses of str should define __slots__ 	🛠️ 🧪
SLOT001 	no-slots-in-tuple-subclass 	Subclasses of tuple should define __slots__ 	🛠️ 🧪
SLOT002 	no-slots-in-namedtuple-subclass 	Subclasses of collections.namedtuple() should define __slots__ 	🛠️ 🧪
flake8-simplify (SIM)#

For more, see flake8-simplify on PyPI.
Code 	Name 	Message 	
SIM101 	duplicate-isinstance-call 	Multiple isinstance calls for {name}, merge into a single call 	🛠️ 🧪
SIM102 	collapsible-if 	Use a single if statement instead of nested if statements 	🛠️ 🧪
SIM103 	needless-bool 	Return the condition {condition} directly 	🛠️ 🧪
SIM105 	suppressible-exception 	Use contextlib.suppress({exception}) instead of try-except-pass 	🛠️ 🧪
SIM107 	return-in-try-except-finally 	Don't use return in try-except and finally 	🛠️ 🧪
SIM108 	if-else-block-instead-of-if-exp 	Use ternary operator {contents} instead of if-else-block 	🛠️ 🧪
SIM109 	compare-with-tuple 	Use {replacement} instead of multiple equality comparisons 	🛠️ 🧪
SIM110 	reimplemented-builtin 	Use {replacement} instead of for loop 	🛠️ 🧪
SIM112 	uncapitalized-environment-variables 	Use capitalized environment variable {expected} instead of {actual} 	🛠️ 🧪
SIM114 	if-with-same-arms 	Combine if branches using logical or operator 	🛠️ 🧪
SIM115 	open-file-with-context-handler 	Use context handler for opening files 	🛠️ 🧪
SIM116 	if-else-block-instead-of-dict-lookup 	Use a dictionary instead of consecutive if statements 	🛠️ 🧪
SIM117 	multiple-with-statements 	Use a single with statement with multiple contexts instead of nested with statements 	🛠️ 🧪
SIM118 	in-dict-keys 	Use key {operator} dict instead of key {operator} dict.keys() 	🛠️ 🧪
SIM201 	negate-equal-op 	Use {left} != {right} instead of not {left} == {right} 	🛠️ 🧪
SIM202 	negate-not-equal-op 	Use {left} == {right} instead of not {left} != {right} 	🛠️ 🧪
SIM208 	double-negation 	Use {expr} instead of not (not {expr}) 	🛠️ 🧪
SIM210 	if-expr-with-true-false 	Remove unnecessary True if ... else False 	🛠️ 🧪
SIM211 	if-expr-with-false-true 	Use not ... instead of False if ... else True 	🛠️ 🧪
SIM212 	if-expr-with-twisted-arms 	Use {expr_else} if {expr_else} else {expr_body} instead of {expr_body} if not {expr_else} else {expr_else} 	🛠️ 🧪
SIM220 	expr-and-not-expr 	Use False instead of {name} and not {name} 	🛠️ 🧪
SIM221 	expr-or-not-expr 	Use True instead of {name} or not {name} 	🛠️ 🧪
SIM222 	expr-or-true 	Use {expr} instead of {replaced} 	🛠️ 🧪
SIM223 	expr-and-false 	Use {expr} instead of {replaced} 	🛠️ 🧪
SIM300 	yoda-conditions 	Yoda conditions are discouraged, use {suggestion} instead 	🛠️ 🧪
SIM401 	if-else-block-instead-of-dict-get 	Use {contents} instead of an if block 	🛠️ 🧪
SIM910 	dict-get-with-none-default 	Use {expected} instead of {actual} 	🛠️ 🧪
flake8-tidy-imports (TID)#

For more, see flake8-tidy-imports on PyPI.

For related settings, see flake8-tidy-imports.
Code 	Name 	Message 	
TID251 	banned-api 	{name} is banned: 	🛠️ 🧪
TID252 	relative-imports 	Relative imports from parent modules are banned 	🛠️ 🧪
TID253 	banned-module-level-imports 	{name} is banned at the module level 	🛠️ 🧪
flake8-type-checking (TCH)#

For more, see flake8-type-checking on PyPI.

For related settings, see flake8-type-checking.
Code 	Name 	Message 	
TCH001 	typing-only-first-party-import 	Move application import {} into a type-checking block 	🛠️ 🧪
TCH002 	typing-only-third-party-import 	Move third-party import {} into a type-checking block 	🛠️ 🧪
TCH003 	typing-only-standard-library-import 	Move standard library import {} into a type-checking block 	🛠️ 🧪
TCH004 	runtime-import-in-type-checking-block 	Move import {qualified_name} out of type-checking block. Import is used for more than type hinting. 	🛠️ 🧪
TCH005 	empty-type-checking-block 	Found empty type-checking block 	🛠️ 🧪
flake8-gettext (INT)#

For more, see flake8-gettext on PyPI.

For related settings, see flake8-gettext.
Code 	Name 	Message 	
INT001 	f-string-in-get-text-func-call 	f-string is resolved before function call; consider _("string %s") % arg 	🛠️ 🧪
INT002 	format-in-get-text-func-call 	format method argument is resolved before function call; consider _("string %s") % arg 	🛠️ 🧪
INT003 	printf-in-get-text-func-call 	printf-style format is resolved before function call; consider _("string %s") % arg 	🛠️ 🧪
flake8-unused-arguments (ARG)#

For more, see flake8-unused-arguments on PyPI.

For related settings, see flake8-unused-arguments.
Code 	Name 	Message 	
ARG001 	unused-function-argument 	Unused function argument: {name} 	🛠️ 🧪
ARG002 	unused-method-argument 	Unused method argument: {name} 	🛠️ 🧪
ARG003 	unused-class-method-argument 	Unused class method argument: {name} 	🛠️ 🧪
ARG004 	unused-static-method-argument 	Unused static method argument: {name} 	🛠️ 🧪
ARG005 	unused-lambda-argument 	Unused lambda argument: {name} 	🛠️ 🧪
flake8-use-pathlib (PTH)#

For more, see flake8-use-pathlib on PyPI.
Code 	Name 	Message 	
PTH100 	os-path-abspath 	os.path.abspath() should be replaced by Path.resolve() 	🛠️ 🧪
PTH101 	os-chmod 	os.chmod() should be replaced by Path.chmod() 	🛠️ 🧪
PTH102 	os-mkdir 	os.mkdir() should be replaced by Path.mkdir() 	🛠️ 🧪
PTH103 	os-makedirs 	os.makedirs() should be replaced by Path.mkdir(parents=True) 	🛠️ 🧪
PTH104 	os-rename 	os.rename() should be replaced by Path.rename() 	🛠️ 🧪
PTH105 	os-replace 	os.replace() should be replaced by Path.replace() 	🛠️ 🧪
PTH106 	os-rmdir 	os.rmdir() should be replaced by Path.rmdir() 	🛠️ 🧪
PTH107 	os-remove 	os.remove() should be replaced by Path.unlink() 	🛠️ 🧪
PTH108 	os-unlink 	os.unlink() should be replaced by Path.unlink() 	🛠️ 🧪
PTH109 	os-getcwd 	os.getcwd() should be replaced by Path.cwd() 	🛠️ 🧪
PTH110 	os-path-exists 	os.path.exists() should be replaced by Path.exists() 	🛠️ 🧪
PTH111 	os-path-expanduser 	os.path.expanduser() should be replaced by Path.expanduser() 	🛠️ 🧪
PTH112 	os-path-isdir 	os.path.isdir() should be replaced by Path.is_dir() 	🛠️ 🧪
PTH113 	os-path-isfile 	os.path.isfile() should be replaced by Path.is_file() 	🛠️ 🧪
PTH114 	os-path-islink 	os.path.islink() should be replaced by Path.is_symlink() 	🛠️ 🧪
PTH115 	os-readlink 	os.readlink() should be replaced by Path.readlink() 	🛠️ 🧪
PTH116 	os-stat 	os.stat() should be replaced by Path.stat(), Path.owner(), or Path.group() 	🛠️ 🧪
PTH117 	os-path-isabs 	os.path.isabs() should be replaced by Path.is_absolute() 	🛠️ 🧪
PTH118 	os-path-join 	os.{module}.join() should be replaced by Path with / operator 	🛠️ 🧪
PTH119 	os-path-basename 	os.path.basename() should be replaced by Path.name 	🛠️ 🧪
PTH120 	os-path-dirname 	os.path.dirname() should be replaced by Path.parent 	🛠️ 🧪
PTH121 	os-path-samefile 	os.path.samefile() should be replaced by Path.samefile() 	🛠️ 🧪
PTH122 	os-path-splitext 	os.path.splitext() should be replaced by Path.suffix, Path.stem, and Path.parent 	🛠️ 🧪
PTH123 	builtin-open 	open() should be replaced by Path.open() 	🛠️ 🧪
PTH124 	py-path 	py.path is in maintenance mode, use pathlib instead 	🛠️ 🧪
PTH201 	path-constructor-current-directory 	Do not pass the current directory explicitly to Path 	🛠️ 🧪
PTH202 	os-path-getsize 	os.path.getsize should be replaced by Path.stat().st_size 	🛠️ 🧪
PTH203 	os-path-getatime 	os.path.getatime should be replaced by Path.stat().st_atime 	🛠️ 🧪
PTH204 	os-path-getmtime 	os.path.getmtime should be replaced by Path.stat().st_mtime 	🛠️ 🧪
PTH205 	os-path-getctime 	os.path.getctime should be replaced by Path.stat().st_ctime 	🛠️ 🧪
PTH206 	os-sep-split 	Replace .split(os.sep) with Path.parts 	🛠️ 🧪
PTH207 	glob 	Replace {function} with Path.glob or Path.rglob 	🛠️ 🧪
flake8-todos (TD)#

For more, see flake8-todos on GitHub.
Code 	Name 	Message 	
TD001 	invalid-todo-tag 	Invalid TODO tag: {tag} 	🛠️ 🧪
TD002 	missing-todo-author 	Missing author in TODO; try: # TODO(<author_name>): ... or # TODO @<author_name>: ... 	🛠️ 🧪
TD003 	missing-todo-link 	Missing issue link on the line following this TODO 	🛠️ 🧪
TD004 	missing-todo-colon 	Missing colon in TODO 	🛠️ 🧪
TD005 	missing-todo-description 	Missing issue description after TODO 	🛠️ 🧪
TD006 	invalid-todo-capitalization 	Invalid TODO capitalization: {tag} should be TODO 	🛠️ 🧪
TD007 	missing-space-after-todo-colon 	Missing space after colon in TODO 	🛠️ 🧪
flake8-fixme (FIX)#

For more, see flake8-fixme on GitHub.
Code 	Name 	Message 	
FIX001 	line-contains-fixme 	Line contains FIXME, consider resolving the issue 	🛠️ 🧪
FIX002 	line-contains-todo 	Line contains TODO, consider resolving the issue 	🛠️ 🧪
FIX003 	line-contains-xxx 	Line contains XXX, consider resolving the issue 	🛠️ 🧪
FIX004 	line-contains-hack 	Line contains HACK, consider resolving the issue 	🛠️ 🧪
eradicate (ERA)#

For more, see eradicate on PyPI.
Code 	Name 	Message 	
ERA001 	commented-out-code 	Found commented-out code 	🛠️ 🧪
pandas-vet (PD)#

For more, see pandas-vet on PyPI.
Code 	Name 	Message 	
PD002 	pandas-use-of-inplace-argument 	inplace=True should be avoided; it has inconsistent behavior 	🛠️ 🧪
PD003 	pandas-use-of-dot-is-null 	.isna is preferred to .isnull; functionality is equivalent 	🛠️ 🧪
PD004 	pandas-use-of-dot-not-null 	.notna is preferred to .notnull; functionality is equivalent 	🛠️ 🧪
PD007 	pandas-use-of-dot-ix 	.ix is deprecated; use more explicit .loc or .iloc 	🛠️ 🧪
PD008 	pandas-use-of-dot-at 	Use .loc instead of .at. If speed is important, use NumPy. 	🛠️ 🧪
PD009 	pandas-use-of-dot-iat 	Use .iloc instead of .iat. If speed is important, use NumPy. 	🛠️ 🧪
PD010 	pandas-use-of-dot-pivot-or-unstack 	.pivot_table is preferred to .pivot or .unstack; provides same functionality 	🛠️ 🧪
PD011 	pandas-use-of-dot-values 	Use .to_numpy() instead of .values 	🛠️ 🧪
PD012 	pandas-use-of-dot-read-table 	Use .read_csv instead of .read_table to read CSV files 	🛠️ 🧪
PD013 	pandas-use-of-dot-stack 	.melt is preferred to .stack; provides same functionality 	🛠️ 🧪
PD015 	pandas-use-of-pd-merge 	Use .merge method instead of pd.merge function. They have equivalent functionality. 	🛠️ 🧪
PD101 	pandas-nunique-constant-series-check 	Using series.nunique() for checking that a series is constant is inefficient 	🛠️ 🧪
PD901 	pandas-df-variable-name 	df is a bad variable name. Be kinder to your future self. 	🛠️ 🧪
pygrep-hooks (PGH)#

For more, see pygrep-hooks on GitHub.
Code 	Name 	Message 	
PGH001 	eval 	No builtin eval() allowed 	🛠️ 🧪
PGH002 	deprecated-log-warn 	warn is deprecated in favor of warning 	🛠️ 🧪
PGH003 	blanket-type-ignore 	Use specific rule codes when ignoring type issues 	🛠️ 🧪
PGH004 	blanket-noqa 	Use specific rule codes when using noqa 	🛠️ 🧪
PGH005 	invalid-mock-access 	Mock method should be called: {name} 	🛠️ 🧪
Pylint (PL)#

For more, see Pylint on PyPI.
Code 	Name 	Message 	
PLC0105 	type-name-incorrect-variance 	{kind} name "{param_name}" does not reflect its {variance}; consider renaming it to "{replacement_name}" 	🛠️ 🧪
PLC0131 	type-bivariance 	{kind} cannot be both covariant and contravariant 	🛠️ 🧪
PLC0132 	type-param-name-mismatch 	{kind} name {param_name} does not match assigned variable name {var_name} 	🛠️ 🧪
PLC0205 	single-string-slots 	Class __slots__ should be a non-string iterable 	🛠️ 🧪
PLC0208 	iteration-over-set 	Use a sequence type instead of a set when iterating over values 	🛠️ 🧪
PLC0414 	useless-import-alias 	Import alias does not rename original package 	🛠️ 🧪
PLC1901 	compare-to-empty-string 	{existing} can be simplified to {replacement} as an empty string is falsey 	🛠️ 🧪
PLC3002 	unnecessary-direct-lambda-call 	Lambda expression called directly. Execute the expression inline instead. 	🛠️ 🧪
PLE0100 	yield-in-init 	__init__ method is a generator 	🛠️ 🧪
PLE0101 	return-in-init 	Explicit return in __init__ 	🛠️ 🧪
PLE0116 	continue-in-finally 	continue not supported inside finally clause 	🛠️ 🧪
PLE0117 	nonlocal-without-binding 	Nonlocal name {name} found without binding 	🛠️ 🧪
PLE0118 	load-before-global-declaration 	Name {name} is used prior to global declaration on line 	🛠️ 🧪
PLE0241 	duplicate-bases 	Duplicate base {base} for class {class} 	🛠️ 🧪
PLE0302 	unexpected-special-method-signature 	The special method {} expects {}, {} {} given 	🛠️ 🧪
PLE0307 	invalid-str-return-type 	__str__ does not return str 	🛠️ 🧪
PLE0604 	invalid-all-object 	Invalid object in __all__, must contain only strings 	🛠️ 🧪
PLE0605 	invalid-all-format 	Invalid format for __all__, must be tuple or list 	🛠️ 🧪
PLE1142 	await-outside-async 	await should be used within an async function 	🛠️ 🧪
PLE1205 	logging-too-many-args 	Too many arguments for logging format string 	🛠️ 🧪
PLE1206 	logging-too-few-args 	Not enough arguments for logging format string 	🛠️ 🧪
PLE1300 	bad-string-format-character 	Unsupported format character '{format_char}' 	🛠️ 🧪
PLE1307 	bad-string-format-type 	Format type does not match argument type 	🛠️ 🧪
PLE1310 	bad-str-strip-call 	String {strip} call contains duplicate characters (did you mean {removal}?) 	🛠️ 🧪
PLE1507 	invalid-envvar-value 	Invalid type for initial os.getenv argument; expected str 	🛠️ 🧪
PLE1700 	yield-from-in-async-function 	yield from statement in async function; use async for instead 	🛠️ 🧪
PLE2502 	bidirectional-unicode 	Contains control characters that can permit obfuscated code 	🛠️ 🧪
PLE2510 	invalid-character-backspace 	Invalid unescaped character backspace, use "\b" instead 	🛠️ 🧪
PLE2512 	invalid-character-sub 	Invalid unescaped character SUB, use "\x1A" instead 	🛠️ 🧪
PLE2513 	invalid-character-esc 	Invalid unescaped character ESC, use "\x1B" instead 	🛠️ 🧪
PLE2514 	invalid-character-nul 	Invalid unescaped character NUL, use "\0" instead 	🛠️ 🧪
PLE2515 	invalid-character-zero-width-space 	Invalid unescaped character zero-width-space, use "\u200B" instead 	🛠️ 🧪
PLR0124 	comparison-with-itself 	Name compared with itself, consider replacing {actual} 	🛠️ 🧪
PLR0133 	comparison-of-constant 	Two constants compared in a comparison, consider replacing {left_constant} {} {right_constant} 	🛠️ 🧪
PLR0206 	property-with-parameters 	Cannot have defined parameters for properties 	🛠️ 🧪
PLR0402 	manual-from-import 	Use from {module} import {name} in lieu of alias 	🛠️ 🧪
PLR0904 	too-many-public-methods 	Too many public methods ({methods} > {max_methods}) 	🛠️ 🧪
PLR0911 	too-many-return-statements 	Too many return statements ({returns} > {max_returns}) 	🛠️ 🧪
PLR0912 	too-many-branches 	Too many branches ({branches} > {max_branches}) 	🛠️ 🧪
PLR0913 	too-many-arguments 	Too many arguments to function call ({c_args} > {max_args}) 	🛠️ 🧪
PLR0915 	too-many-statements 	Too many statements ({statements} > {max_statements}) 	🛠️ 🧪
PLR1701 	repeated-isinstance-calls 	Merge isinstance calls: {expression} 	🛠️ 🧪
PLR1711 	useless-return 	Useless return statement at end of function 	🛠️ 🧪
PLR1714 	repeated-equality-comparison 	Consider merging multiple comparisons: {expression}. Use a set if the elements are hashable. 	🛠️ 🧪
PLR1722 	sys-exit-alias 	Use sys.exit() instead of {name} 	🛠️ 🧪
PLR2004 	magic-value-comparison 	Magic value used in comparison, consider replacing {value} with a constant variable 	🛠️ 🧪
PLR5501 	collapsible-else-if 	Use elif instead of else then if, to reduce indentation 	🛠️ 🧪
PLR6301 	no-self-use 	Method {method_name} could be a function or static method 	🛠️ 🧪
PLW0120 	useless-else-on-loop 	else clause on loop without a break statement; remove the else and de-indent all the code inside it 	🛠️ 🧪
PLW0127 	self-assigning-variable 	Self-assignment of variable {name} 	🛠️ 🧪
PLW0129 	assert-on-string-literal 	Asserting on an empty string literal will never pass 	🛠️ 🧪
PLW0131 	named-expr-without-context 	Named expression used without context 	🛠️ 🧪
PLW0406 	import-self 	Module {name} imports itself 	🛠️ 🧪
PLW0602 	global-variable-not-assigned 	Using global for {name} but no assignment is done 	🛠️ 🧪
PLW0603 	global-statement 	Using the global statement to update {name} is discouraged 	🛠️ 🧪
PLW0711 	binary-op-exception 	Exception to catch is the result of a binary and operation 	🛠️ 🧪
PLW1508 	invalid-envvar-default 	Invalid type for environment variable default; expected str or None 	🛠️ 🧪
PLW1509 	subprocess-popen-preexec-fn 	preexec_fn argument is unsafe when using threads 	🛠️ 🧪
PLW1510 	subprocess-run-without-check 	subprocess.run without explicit check argument 	🛠️ 🧪
PLW1641 	eq-without-hash 	Object does not implement __hash__ method 	🛠️ 🧪
PLW2901 	redefined-loop-name 	Outer {outer_kind} variable {name} overwritten by inner {inner_kind} target 	🛠️ 🧪
PLW3201 	bad-dunder-method-name 	Bad or misspelled dunder method name {name}. (bad-dunder-name) 	🛠️ 🧪
PLW3301 	nested-min-max 	Nested {func} calls can be flattened 	🛠️ 🧪
tryceratops (TRY)#

For more, see tryceratops on PyPI.
Code 	Name 	Message 	
TRY002 	raise-vanilla-class 	Create your own exception 	🛠️ 🧪
TRY003 	raise-vanilla-args 	Avoid specifying long messages outside the exception class 	🛠️ 🧪
TRY004 	type-check-without-type-error 	Prefer TypeError exception for invalid type 	🛠️ 🧪
TRY200 	reraise-no-cause 	Use raise from to specify exception cause 	🛠️ 🧪
TRY201 	verbose-raise 	Use raise without specifying exception name 	🛠️ 🧪
TRY300 	try-consider-else 	Consider moving this statement to an else block 	🛠️ 🧪
TRY301 	raise-within-try 	Abstract raise to an inner function 	🛠️ 🧪
TRY302 	useless-try-except 	Remove exception handler; error is immediately re-raised 	🛠️ 🧪
TRY400 	error-instead-of-exception 	Use logging.exception instead of logging.error 	🛠️ 🧪
TRY401 	verbose-log-message 	Redundant exception object included in logging.exception call 	🛠️ 🧪
flynt (FLY)#

For more, see flynt on PyPI.
Code 	Name 	Message 	
FLY002 	static-join-to-f-string 	Consider {expression} instead of string join 	🛠️ 🧪
NumPy-specific rules (NPY)#
Code 	Name 	Message 	
NPY001 	numpy-deprecated-type-alias 	Type alias np.{type_name} is deprecated, replace with builtin type 	🛠️ 🧪
NPY002 	numpy-legacy-random 	Replace legacy np.random.{method_name} call with np.random.Generator 	🛠️ 🧪
NPY003 	numpy-deprecated-function 	np.{existing} is deprecated; use np.{replacement} instead 	🛠️ 🧪
Airflow (AIR)#

For more, see Airflow on PyPI.
Code 	Name 	Message 	
AIR001 	airflow-variable-name-task-id-mismatch 	Task variable name should match the task_id: "{task_id}" 	🛠️ 🧪
Perflint (PERF)#

For more, see Perflint on PyPI.
Code 	Name 	Message 	
PERF101 	unnecessary-list-cast 	Do not cast an iterable to list before iterating over it 	🛠️ 🧪
PERF102 	incorrect-dict-iterator 	When using only the {subset} of a dict use the {subset}() method 	🛠️ 🧪
PERF203 	try-except-in-loop 	try-except within a loop incurs performance overhead 	🛠️ 🧪
PERF401 	manual-list-comprehension 	Use a list comprehension to create a transformed list 	🛠️ 🧪
PERF402 	manual-list-copy 	Use list or list.copy to create a copy of a list 	🛠️ 🧪
PERF403 	manual-dict-comprehension 	Use a dictionary comprehension instead of a for-loop 	🛠️ 🧪
refurb (FURB)#

For more, see refurb on PyPI.
Code 	Name 	Message 	
FURB113 	repeated-append 	Use {suggestion} instead of repeatedly calling {name}.append() 	🛠️ 🧪
FURB131 	delete-full-slice 	Prefer clear over deleting a full slice 	🛠️ 🧪
FURB132 	check-and-remove-from-set 	Use {suggestion} instead of check and remove 	🛠️ 🧪
FURB140 	reimplemented-starmap 	Use itertools.starmap instead of the generator 	🛠️ 🧪
FURB145 	slice-copy 	Prefer copy method over slicing 	🛠️ 🧪
FURB148 	unnecessary-enumerate 	enumerate value is unused, use for x in range(len(y)) instead 	🛠️ 🧪
flake8-logging (LOG)#

For more, see flake8-logging on PyPI.
Code 	Name 	Message 	
LOG001 	direct-logger-instantiation 	Use logging.getLogger() to instantiate loggers 	🛠️ 🧪
LOG002 	invalid-get-logger-argument 	Use __name__ with logging.getLogger() 	🛠️ 🧪
LOG007 	exception-without-exc-info 	Use of logging.exception with falsy exc_info 	🛠️ 🧪
LOG009 	undocumented-warn 	Use of undocumented logging.WARN constant 	🛠️ 🧪
Ruff-specific rules (RUF)#
Code 	Name 	Message 	
RUF001 	ambiguous-unicode-character-string 	String contains ambiguous {}. Did you mean {}? 	🛠️ 🧪
RUF002 	ambiguous-unicode-character-docstring 	Docstring contains ambiguous {}. Did you mean {}? 	🛠️ 🧪
RUF003 	ambiguous-unicode-character-comment 	Comment contains ambiguous {}. Did you mean {}? 	🛠️ 🧪
RUF005 	collection-literal-concatenation 	Consider {expression} instead of concatenation 	🛠️ 🧪
RUF006 	asyncio-dangling-task 	Store a reference to the return value of asyncio.{method} 	🛠️ 🧪
RUF007 	pairwise-over-zipped 	Prefer itertools.pairwise() over zip() when iterating over successive pairs 	🛠️ 🧪
RUF008 	mutable-dataclass-default 	Do not use mutable default values for dataclass attributes 	🛠️ 🧪
RUF009 	function-call-in-dataclass-default-argument 	Do not perform function call {name} in dataclass defaults 	🛠️ 🧪
RUF010 	explicit-f-string-type-conversion 	Use explicit conversion flag 	🛠️ 🧪
RUF011 	static-key-dict-comprehension 	Dictionary comprehension uses static key: {key} 	🛠️ 🧪
RUF012 	mutable-class-default 	Mutable class attributes should be annotated with typing.ClassVar 	🛠️ 🧪
RUF013 	implicit-optional 	PEP 484 prohibits implicit Optional 	🛠️ 🧪
RUF015 	unnecessary-iterable-allocation-for-first-element 	Prefer next({iterable}) over single element slice 	🛠️ 🧪
RUF016 	invalid-index-type 	Slice in indexed access to type {value_type} uses type {index_type} instead of an integer. 	🛠️ 🧪
RUF017 	quadratic-list-summation 	Avoid quadratic list summation 	🛠️ 🧪
RUF100 	unused-noqa 	Unused noqa directive 	🛠️ 🧪
RUF200 	invalid-pyproject-toml 	Failed to parse pyproject.toml: 	🛠️ 🧪
Made with Material for MkDocs Insiders
