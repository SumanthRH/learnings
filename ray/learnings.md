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
- *Task* - A remote function invocation. This is a single function invocation that executes on a process different from the caller, and potentially on a different machine. 
- *Object* - An application value. These are values that are returned by a task or created through `ray.put`. Objects are immutable! 
    - Ray also doesn't handle deduplication. More on this later.
- 
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