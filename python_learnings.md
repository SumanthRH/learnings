# Learnings
These are some notes and comments I've taken while going through the [Advanced Python Mastery](https://github.com/dabeaz-course/python-mastery) course. Many sentences here are copied verbatim from David's slides since I found them to be great summaries for different concepts.

## Table of Contents

<!-- toc -->

- [Python Essentials](#python-essentials)
- [Classes](#classes)
- [Python Encapsulation](#python-encapsulation)
- [Inheritance](#inheritance)
- [Advanced Inheritance and Mixins](#advanced-inheritance-and-mixins)
- [Inside Python Objects](#inside-python-objects)
- [Functions](#functions)
  * [Concurrency and Futures](#concurrency-and-futures)
  * [Functional Programming](#functional-programming)
  * [Closures](#closures)
  * [Exception Handling](#exception-handling)
- [Testing](#testing)
  * [unittest module](#unittest-module)
- [Working with Code](#working-with-code)
  * [Advanced function usage](#advanced-function-usage)
  * [Scoping rules](#scoping-rules)
  * [Function Objects](#function-objects)
  * [`eval` and `exec`](#eval-and-exec)
  * [Callable Objects](#callable-objects)
- [Metaprogramming](#metaprogramming)
  * [Decorator](#decorator)
  * [Class Decorators](#class-decorators)
  * [Types](#types)
  * [Metaclasses](#metaclasses)
- [Iterators, Generators and Coroutines](#iterators-generators-and-coroutines)
  * [Generator Pipelines](#generator-pipelines)
  * [Coroutine](#coroutine)
  * [Generator Control Flow and Managed Generators](#generator-control-flow-and-managed-generators)
- [Modules and Packages](#modules-and-packages)

<!-- tocstop -->

## Python Essentials
- Exercise 2.1: Reading data as a single string vs. reading data with f.readlines() has a huge difference in memory consumption - 10 MB vs 40 MB for an example file. 
Point to ponder: what might be the source of that extra overhead?
    - > My guess is the additional information stored in a list data structure. 

- Memory efficiency:
    - Class with `__slots__` < Tuple < Namedtuple < Class < Dict

- A ChainMap class is provided for quickly linking a number of mappings so they can be treated as a single unit. It is often much faster than creating a new dictionary and running multiple update() calls.

- `zip(list1, list2)` Truncates to shortest input length

- Generators summarized in a line: Generators are useful in contexts where the result is an intermediate step

- You can directly pass in generators in function arguments:
    ```
    sum(x*x for x in nums)
    ```
- Generator expressions can save a LOT of memory. Key idea is that if you're dealing with map and reduce like operations on your data, you can chain operations on generator expressions and then actually materialize results one by one when you do a reduce op like sum(), max(), etc. See ex 2.3 for details - memory usage here goes from ~ 200 MB to ~ 120 KB.

- Use the `dis` library in python to peek into low level byte code. For example, use `dis.dis(f)` to get the byte code for a function `f`

- Builtin types operate according to
predefined "protocols" - the name for the special methods like `__add__` and `__len__`. Object protocols are baked into the interpreter as low-level bytecode 

- Container objects only hold references
(pointers) to their stored values. All operations involving the container internals only manipulate the pointers (not the objects)

    ![container](container.png)

- All "hashable" objects in python have a `__hash__()` and `__eq__()` method

- Assignment operations never make a copy of the value being assigned - all assignments are merely reference copies

- Immutable values can be safely shared, which can save a lot of memory (think a long list of dictionaries)

- `copy.deepcopy` is the only safe way to copy an object in python

## Classes 

- Object oriented programming is largely
concerned with the modeling of "behavior." 

- A class is a simply a set of functions that do different things on "instances"

- Classes do not define a scope. If want to operate on an instance, you always have to refer to it explicitly

- There are only three operations on an instance: 
```
obj.attr # Get an attribute
obj.attr = value # Set an attribute
del obj.attr # Delete an attribute - obj.attr no longer exists
```
- Method calls are layered onto the machinery used for simple attributes.
```
s.get_something # Looks up the method
s.get_something() # looks up and calls the method
```

- Internally, a method call `s.method()` is implemented as `s.method.__func__(s.method.__self__)`

- Class variables: can be accessed at the class level or by an instance. Can also be changed via inheritance.

- Class Method is a method that operates on the class itself. It's invoked on the class, not an instance. Example usecase is in providing alternate constructors. Most popular example of this is `AutoModel.from_pretrained` in 🤗Transformers.

- Implicit conversion of data in `__init__()` can limit flexibility and might introduce weird bugs if a user isn't paying careful attention.

## Python Encapsulation
- Python has programming conventions to indicate the intended use of something. Ex. private and public methods aren't exactly enforced like in C++.
- Attributes with a single underscore "_" are meant to act as private attributes (still accessible in instanecs and for subclasses)
- Attributes with a double underscore "__" have a special meaning - these are not accessible to subclasses. They are acessible in instance via Python's name mangling trick Ex:
```
s = MyClass()
s._MyClass__attr # to get __attr
```
- Properties in python: Useful alternative to accessor methods. Can also make sure a property is not stale/ computed when accessed. Note that a property is a _class variable_.
- Property decorators:
    - @property: to declare a property
    - @<\pname>.setter: function that's invoked with assignment ops. Needed to modify a property
    - @<\pname>.deleter: function that's invoked with deletion

- Advice on `__slots__`:  Do not use it except with classes that are simply going to serve as simple data
structures.

## Inheritance
- Inheritance in a nutshell: Extend existing code. That's it.
    - `__init__` inheritance: you must also initialize the parents with `super().__init__()`
    - objects defined via inheritance are a special version of the parent i.e same capabilities
    - `object` is a parent of all classes in Python 3 (even if not specified)
- Objects have two string representations:
    - str(obj) - printable output
    - repr(obj) - for programmers
    - The convention for `__repr__` is to return a string that, when fed to `eval()` , will recreate the underlying object.
    - print(obj) uses `__str__`
- Item access: `__setitem__`, `__getitem__`, `__delitem__`, `__contains__`, `__len__`

- Instances under the hood:
    ```
    d = Date.__new__(Date, 2012, 12, 21)
    d.__init__(2012, 12, 21)
    ```
- `__del__` : destructor method. Called when the reference count reaches 0. It's not the same as the del operator.`del` decreases ref count by 1

- Weak Reference: A reference to an object that does not increase its reference count

- Context managers: To be used with resources! That's it.
`__enter__()` and `__exit__()`

- Handler classes:  Code will implement a general purpose algorithm, but will defer certain steps to a separately supplied handler object (like the formatter implemented in 3.5)
- Be careful: `isinstance` vs `issubclass`

## Advanced Inheritance and Mixins 
- _Inheritance is a tool for code reuse_

- _Python uses "cooperative multiple inheritance"_

- If you have a class `Child(A, B)`, then the common methods of A and B can get "merged" when accessed via a Child instance! 
    - `super()` moves to the next class in the list of parents

- **Mixin Classes:** A mixin is a class whose purpose is to add extra functionality to other class definitions. For example, the creator of a library can provide a set of classes. Mixins are a collection of add-on classes that can be provided to make those classes behave in different ways.

## Inside Python Objects
- Dictionaries are used for critical parts of the interpreter and may be the most important type of data in Python.
- Each instance gets its own private dictionary: `obj.__dict__`
- "...the entire object system is mostly
just an extra layer that's put on top of
dictionaries.."
- The instance dictionary (`__dict__`) holds data unique to
each instance whereas the class dictionary (`__class__`)
holds data collectively shared by all instances
- When you read an attribute, the attribute might be sitting in a local instance dictionary or the class dictionary: both might be checked (local first, then class)
- class dictionary : access via instance - `obj.__class__` or directly via class `cls.__dict__`
- A class is just a dictionary
- With inheritance, the inheritance chain is precomputed and stored in an "MRO" (Method Resolution Object) attribute on the class - `cls.__mro__`
- `super()` delegates to the next class on the MRO
- ` super()` might not go to the
immediate parent. It's different from doing `parent_cls.attr`
- Designing for inheritance:

    1. Rule 1: Compatible Method Arguments
    2. Rule 2: Method chains must terminate
    3. Rule 3: use `super()` everywhere
        - If multiple inheritance is used, a direct parent call will probably violate the MRO
- Attribute binding: An extra processing step while accessible attributes of classes.
    - When an attribute `cls.attr` is accessed on a class, the attribute is checked to see if it holds a _descriptor_ object. 
    - A descriptor is just an object with get, set and delete methods
    -  Every major feature of classes is implemented using descriptors
    - Functions/Methods are descriptors where `__get__()` creates the bound method object
    - Descriptors are one of Python's most powerful customizations (you own the dot) - you get low level control over the dot and can use it to do amazing things.
- Attribute Access: 
    - When you do `obj.x` -> first, `obj.__getattribute__(x)` is called. This looks for descriptors, checks the instance dictionary, checks bases classes (inheritance), etc.
    If still not found, it will invoke `obj.__getattr__(x)`

## Functions

- Basic design principle: Make functions "self-contained". Avoid hidden side effects. Only operate on passed arguments. Two goals: Simplicity and Predictability. 
- Prefer keyword arguments while passing optional arguments! 
    - Can also force it with * : `read_data(filename, *, debug=False)`
- Don't use mutable values as defaults! Default values are created only once for the whole program.
- Argument Transforms: Design for flexibility
- Doc Strings: feeds the `help()` command and dev tools
- Type Hints: Useful for code checkers, documentation, etc.
- **Return Type**: Have the function cleanly return one result. Just make it a tuple if you really need to

### Concurrency and Futures
- Functions can execute concurrently in separate threads. They'll have a shared state, with execution in a single interpreter. Recall lessons from your Operating Systems course: a thread is a single sequential path of execution in a program.
- Futures: Represents a future result to be computed. 

### Functional Programming
- Callback function: A function passed into another function as an argument, invoked as an action/routine inside the other function.
- Lambda functions: anonymous functions created on the spot
- Lambdas can be used to alter function args similar to `functools.partial`
### Closures
-  If an inner function is returned as a result, the inner function is known as a "closure".
- Variables used are accesssible via the `__closure__` special method
- Only variables that are needed are kept
- Closure variables are also mutable! Can be used to keep mutable internal state
- Applications:
    - Alternate evaluation (e.g., delayed evaluation)
    - Callback functions- what we saw before
    - Code creation ("macros")
### Exception Handling
- What exceptions to handle? Well, when recovery is possible
- Never catch all exceptions
- Wrapping an Exception:
    - `raise TaskError('It failed') from e`
- Don't use return codes!
- Use `AttributeError` when trying to say that an attribute shouldn't be set/ accessed.

## Testing
- Testing rocks, debugging sucks: Python is dynamic, interpreted - there's no compiler to catch your bugs
- Assertions/ Contracts: Assertions are runtime checks. Asserts are meant for program invariants, not to validate user inputs!
    - Can disable via `-O`: `python -O main.py`
### unittest module
- First, create a separate file for testing. Then define test classes like `TestMyFunc(unittest.TestCase)` (note that they must inherit from TestCase). 
- Define testing methods - check method should start with `test`
- Each test uses special assertions like `assertTrue`, `assertFalse`
- To run the test, just add `unittest.main()` in the main block. Can now run with `python`!

## Working with Code
### Advanced function usage
- Some simple stuff first: 
    - `func(*args)` accepts any number of arguments
    - function that accepts any keyword arguments: `func(x, **kwargs)` => extra keywords get passed in a dict
    - function that accepts any arguments: `func(*args, **kwargs)` `args` captures positional, `kwargs` captures keyword
### Scoping rules
- All statements have 2 scopes/ access to two scopes: The global scope of the module in which it is in, and the local scope, the scope private to the function it is in
- Global variables can be accessed within a function, but it cannot be modified without an explicit `global` keyword. This is important: Globals are already readable! Only before you modify them, you need to specify this keyword
- `globals()`/`locals()`: Gives you a dictionary with the contents of the global/local scope respectively
- Scope of the built-ins: The built-ins like `abs()`, `repr()` are in a special module called `builtins`. You can even modify it here (but is ill-advised)
- Frame Hacking: You can move up the stack frame with `sys._getframe()`. For example, to use the local variables of the caller inside a function, you can do `sys._getframe(1).f_locals`

### Function Objects
- Functions are also objects in python. You can pass them around, assign them to variables, and also inspect attributes. 
- Docstring: The first line of a function can be a string. Inspect via `func.__doc__`
- Type hints/ annotations: Inspect via `func.__annotations__`. Of course, as we are all painfully aware, type annotations do absolutely nothing in Python.
- You can also just add random attributes to a function! Stored in `func.__dict__`.
- More helpful attributes: `func.__defaults__`, `func.__code__.co_argcount`, `func.__code__.co_varnames`
- `inspect` module in python: enables more structured inspection.

### `eval` and `exec`
- `eval`: Evaluates arbitrary expressions (like `3x**2 + 2`)
- `exec`: Executes arbitrary statements (like `print(x)`)
- `exec`: Modifications to local scope are lost! `exec(x=10; print(x))` doesn't modify `x` in locals()
- To make sure modifications last, exec also gives optional arguments: `exec(code [,globals [,locals]])`
- Both of these are meant to be used with extreme care because, well, the expressions and statements can be arbitrary (security??) and interaction with scoping/variables is tricky.
- Python's `namedtuple` uses `eval` internally! (Correction in pdf: it's `eval`, not `exec`)

### Callable Objects
- Any object with a `__call__` special method (emulates functions)
- Signature binding is lovely:
```
sig = inspect.signature(self.func)
bound_sig = sig.bind(*args, **kwargs) # Binds the passed args and kwargs to function arguments
for name, val in bound_sig.arguments.items():
    <do things>
```
- There are caveats when you use signature binding directly with methods! (Because of the `self` argument)

##  Metaprogramming 
- Metaprogramming pertains to the problem of writing code that manipulates other code. 
    - Macros, Wrappers, Aspects, etc
### Decorator
- Decorator: Function that wraps another function
    ```
    def decorator(func):
        def wrapped(*args, **kwargs):
            .... do things ....
            return func(*args', **kwargs')
        return wrapped
    ```
- Typically, you want to continue using the same function that has been "wrapped" by another function. The function has not been "decorated" with extra features.
- Use a decorator anytime you want to define a kind of "macro" involving function definitions. Ex: execution time
- Decorators with arguments: The decorator returns a function that accepts arguments and then returns...a wrapper!
### Class Decorators
- Class decorators are similar to function decorators: they take in a class and return a class. However, a key difference is that, they return the original class itself, instead of returning a wrapped version like what function decorators do.
    ```
    def class_decorator(cls):
        ... do things ...
        return cls
    ```
- Decoration via Inheritance: base classes can use the `__init_subclass__` special method to observe inheritance and inject some behaviour. HF uses this to propagate warnings to [subclasses](https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/models/bart/modeling_bart.py#L871).

### Types
- All values have an associated type in python `type(var)`. The "type" returned is usually a callable that can make values of that type. `int('1')`.
- Classes define new types. The class is the type for new instances created. THe class is also a callable to create instances of that type.
- Classes are instances of `type`! `type(cls)` returns `<class type>`
- Types are represented by their own classes - `type`! This creates new type objects!
- The internals of class definitions:
    1. The body of the class is captured as a string
    2. A dictionary is created (`.__dict__`) with some metadata
    3. Class body is executed in the dict `exec(body, globals(), __dict__)`
    4. Class is constructed as `type(name, bases, dict)`
- The class is nothing more than this: The name, the base classes it's inheriting from, and the dictionary!

### Metaclasses
- A class that creates classes is called a metaclass.
- Python provides a metaclass hook that allows you to override the class creation steps 
    - `metaclass` keyword argument (`class Spam(metaclass=type)`) to specify the class to create the class. By default, it is `type` for all classes
    - New `metaclass`: Inherit from `type` and customize `__new__`, `__prepare__`. Then, create a new root-object (similar to `object`). Now, you can inherit from this.
- Class creation:
    ```
    type.__prepare__(name, bases)
    type.__new__(type, name, bases, dict)
    type.__init__(cls, name, bases, dict)
    ```
- Instance creation:
    ```
    type.__call__(cls, *args, **kwargs)
    ```
- 🤗Accelerate uses metaclasses to add a [key handler](https://github.com/huggingface/accelerate/blob/649e65b542a5740fb5ce663bbd5af45ed426c06f/src/accelerate/commands/menu/input.py#L53) method to different classes that accept keyboard inputs. 

## Iterators, Generators and Coroutines
- Iterators: have the `__iter__()` and `__next__()` methods
- Generators: simplify custom iteration
- Generator functions: wacky
    - Calling a generator function creates a generator object. It does not start running the function.
    - Function only executes on `next()`
    - Generators are one time use. 
- Resuable generators: Make a class with `__iter__()` that is a generator function! Every use of `__iter__()` makes a new generator
- Want a custom iterator? Always go for a generator
- Python uses `__iter__` for unpacking as well! Ex:  `arg1, arg2, arg3 = obj`

### Generator Pipelines
- Producer-> Processors -> Consumer
- Intermediate processing stages simultaneously
consume and produce items. These can modify the data items, filter/discard items, etc

### Coroutine
- `x = yield`: you get a coroutine! This is a function to which you can send values.
- `func.send("hey")`: sent values are returned by `yield` and execution continues from there. 
- Coroutines are similar to generators: When you call one, nothing happens. All coroutines must be primed first with  with `.send(None)`. This lands execution at the location of the first yield expresssion. At this point, it can receive a value!
- You can define intermediate stages/ processors as coroutines! They have a `yield` and also call `.send()` on the next stage!
### Generator Control Flow and Managed Generators
- Generators have support for forced termination (`.close()`) and exception handling (`.throw()`)
    - `.close()` raises `GeneratorExit` at `yield`
    - `.throw(Error, "error str")` raises an exception at `yield`
- A generator function cannot execute by itself (need a for loop or a send())
- The `yield` statement represents the point of preemption. This is where executed last stopped
- Managed generators: A manager coordinates the execution of a collection of generators.
    - Concurrency, Actors, Event simulation
- Python provides a handy `yield from` syntax to delegate generation (i.e writing the for loop/`send()`) to outer code that calls this.
- Async: `await` just looks like alternate syntax for coroutines (`send`)

## Modules and Packages
- The very basic: Every source file is a module. The import module will load AND execute a module.
- Modules are a namespace for the definitions inside, and as expected, a layer over a Python dictionary (the globals of that module).
- Special variables: `__file__` (name of source file), `__name__` (name of the module) and `__doc__`
    - THe main module: `__name__ == __main__` 
- `import foo` in `bar.py`:  Executes `foo` and adds a reference to `foo` in `bar.__dict__`
    - `from foo import func` : `foo` still executes, but only `func` is added to `bar`'s dictionary 
    - `from foo import *`: All names that don't start with underscore get added to `bar`
- Each module is loaded only once. Use `sys.modules` to get a list of loaded modules.
- Module reloading (with `importlib`): This is not advised, since existing instances of classes will still use old code, specific names imported `from foo import name` don't get updated, and can break typechecks/ code with `super()`
- Module import basics: Relative imports of submodules don't work - imports are always absolute
    - Use "." prefix
- Packages: `__package__` (name of the enclosing package), `__path__` (search path for subcomponents)
- Main use of `__init__.py`: stitching together multiple source files into a "unified" top-level import
- Controlling exports: Submodules should define `__all__` to control `from module import *`. This is useful in `__init__`!
- `__main__.py` designates an entrypoint for a package! Makes the module executable `python -m module`
- Inject dynamic import magic with `__import__` or `importlib`. Can do things like `__import__(f"{__package__}.formats.{format}")` (this imports a file from `.formats` if present)