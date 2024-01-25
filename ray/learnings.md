This is a summary of my learnings on Ray. A lot of the wordings here are borrowed verbatim from the [Ray architecture whitepaper](https://docs.google.com/document/d/1tBw9A4j62ruI5omIJbMxly-la5w4q_TjyJgJL_jN2fI/preview). There's really no greater resource than the document. I am also going through the Learning Ray book from Anyscale, which is good for applications but not for internals.

# Table of Contents

<!-- toc -->

- [The basics](#the-basics)
- [Major anti-pattern](#major-anti-pattern)
- [Actors](#actors)
  * [Execution](#execution)
  * [Failure Model](#failure-model)

<!-- tocstop -->

# The basics
- *Task* - A remote function invocation (an instance of a `@ray.remote` function). This is a single function invocation that executes on a process different from the caller, and potentially on a different machine. 
- *Object* - An application value. These are values that are returned by a task or created through `ray.put`. Objects are immutable! 
    - Ray also doesn't handle deduplication. More on this later.
- *Actor* - a stateful worker process (an instance of a `@ray.remote` class). Actor tasks must be submitted with a handle, or a Python reference to a specific instance of an actor, and can modify the actor’s internal state during execution.
- Driver - The program root, or the “main” program. This is the code that runs `ray.init()`
    - TODO: What about a multi-node setup? I'm guessing ray.init is run only by the head node?
- Job - The collection of tasks, objects, and actors originating (recursively) from the same driver, and their runtime environment. There is a 1:1 mapping between drivers and jobs.
    - A job is a *collection*. It is not a single task/ function invocation. Each driver is executing a "job".
# Design
This part is pretty complicated at first glance. You can slowly start to see why this makes sense by going over more and more examples and applications of Ray. 

![Alt text](ray_design.png)

- Raylet: This is shared within a node! Notice how each node has one raylet, which has two components:
    - Scheduler: Responsible for resource management, task placement, and fulfilling task arguments that are stored in the distributed object store. (Basically, the scheduler in each node acts when new tasks are launched and object references are resolved/ derefenced)
    - Shared-memory object store: This is the shared object store between all workers in a node. This is where large objects get stored.

Ray, overall, has a distributed object store with a decentralized task scheduling system. (the big words come in because each node can do the thing on it's own/ has it's own thing)

# Memory model
To understand how things work here, it is probably better to get a quick refresher on memory allocation in Python and C++
## Quick Refersher for Python and C++
- In C++, all local variables are, by default, allocated on the stack (unless you use `malloc`).  Recall that each function gets it's own stack (the "stack frame") where it stores arguments, local variables and return addresses. The stack is LIFO memory.
- `malloc` is used for dynamically allocating memory. This is memory allocated on the heap. Heap memory has to be managed manually by the user (while stack memory is managed automatically with function returns/program exit). 
- Python manages memory with a private heap that stores all objects and data structures. This is not because Python is interpreted, by the way. There are optimizations you can do to avoid heap allocation by default, and you can read up on PostScript. But, in summary, Python's memory manager allocates all objects on the heap, and, instead of manual management by the user, we have Python's infamous garbage collector toiling behind the scenes to clean up our mess.
## Ray's memory model

![Alt text](ray_memory.png)

There's a lot going on in the about picture. I frankly prefer the simple list of different memory components:
- <u>Heap memory used by Ray workers during task or actor execution</u>: This is, to simplify things a bit, like your regular Python memory management on the heap. Note that the user/ application developer needs to be mindful of excessive memory usage due to a large number of workers, etc.
- <u>Shared memory used by large Ray objects</u>: This is shared memory used when you do a `ray.put` call, or those returned by a Ray task!
    - **Important:** The return values of Ray tasks are stored in the object store. Contrast this with regular local memory that is used by a task, which gets allocated on the heap
    - **Garbage collection:** We were just discussing about garbage collection on heap/stack. Well, how does garbage collection work for Ray's shared memory object store? Ray does this for you via reference counts (much like Python's garbage collector)
- <u>Heap memory used by small Ray objects</u>:

# Major anti-pattern

# Actors
When an actor is created in Python, the creating worker builds a special task known as an actor creation task that runs the actor’s Python constructor.

All actor data is centrally stored and managed by the GCS service. 
Actor creation is non-blocking - the Python call to create the actor immediately returns an “actor handle” that can be used even if the actor creation task has not yet been scheduled

## Execution
Similar to tasks, they return futures, are submitted directly to the actor process via gRPC, and will not run until all `ObjectRef` dependencies have been resolved. 
- For each caller of an actor, the tasks are executed in the same order that they are submitted. This is because the tasks are assumed to modify the actor state.
- Ray provides `async actors` if you want actor tasks to be run concurrently

## Failure Model
- All nodes are assigned a unique identifier and communicate with each other through heartbeats.