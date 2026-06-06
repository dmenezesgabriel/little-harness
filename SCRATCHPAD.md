Explore my codebase and assess whether it has stable abstractions, interfaces, and semantic boundaries. Check whether OOP is used consistently where appropriate, and whether SOLID principles are respected: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion.

Evaluate whether polymorphism is used correctly, whether Jeff Bay’s Object Calisthenics are mostly respected, and whether external dependencies are isolated behind clear boundaries. Check whether classes are behavior-rich and not anemic data containers, with domain rules placed close to the data they protect.

Check naming conventions for idiomatic industry/open-source standards, clarity, consistency, and domain meaning. Review whether the directory structure follows community conventions and clearly separates core domain, application logic, infrastructure, adapters, and extensions.

Verify whether the codebase makes good use of standard libraries and their built-in benefits before introducing custom code or extra dependencies.

Assess memory and resource management. Identify opportunities for generators, context managers, streaming, caching, thread-safe parallelism, async concurrency, or simpler sequential execution where appropriate.

Review whether observability is intrinsic and meaningful, with structured logging, tracing, metrics, useful context, correlation IDs, and actionable error messages.

Verify whether linters, formatters, type checks, boundary guardrails, dependency checks, security checks, test suites, and CI commands are aligned with the current architecture, directory structure, and real project risks. Ensure strong typing is enforced, not bypassed: no unnecessary `any`, casts, ignores, unchecked dynamic access, untyped public APIs, or type-check suppression unless justified and documented.

Identify whether any design pattern would simplify or stabilize the architecture, such as Facade, Strategy, Factory, Proxy, Adapter, State, or Observer. Avoid adding patterns unless they reduce coupling, duplication, or instability.

Following the existing patter I want from now to separate what is core from what should be an extension/plugin thatones that you install with uv pip install package[package]

Return findings with evidence from the codebase, risks, and concrete refactoring recommendations prioritized by impact.


---

Return findings with evidence from the codebase, risks, and concrete refactoring recommendations prioritized by impact.

----

for example I would like to try using a litellm provider with my harness now to check if would work nicely but I want it to be plug and play, and also Llamacpp must be plug and play because they are external. I considered astral-uv monorepo approach with plugins because lateron several things will be plugable and replaceble like memory, other tools. what is your recommendation?


---

Following the existing patterns choose the best way to implement hooks like SessionStart, UserPromptSubmit, PreToolUse, PostToolUse, Stop, SessionEnd

---
Return findings with evidence from the codebase, risks, and concrete refactoring recommendations prioritized by impact.

---

Explore my codebase and implement the below feature using stable abstractions, interfaces, and semantic boundaries. Ensure OOP is used consistently where appropriate, and that SOLID principles are respected: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, and Dependency Inversion.

Ensure polymorphism is used correctly, Jeff Bay’s Object Calisthenics are mostly respected, and external dependencies are isolated behind clear boundaries. Ensure classes are behavior-rich and not anemic data containers, with domain rules placed close to the data they protect.

Apply naming conventions aligned with idiomatic industry/open-source standards, clarity, consistency, and domain meaning. Adjust the directory structure to follow community conventions and clearly separate core domain, application logic, infrastructure, adapters, and extensions.

Ensure the codebase makes good use of standard libraries and their built-in benefits before introducing custom code or extra dependencies.

Improve memory and resource management. Apply generators, context managers, streaming, caching, thread-safe parallelism, async concurrency, or simpler sequential execution where appropriate.

Make observability intrinsic and meaningful, with structured logging, tracing, metrics, useful context, correlation IDs, and actionable error messages.

Ensure linters, formatters, type checks, boundary guardrails, dependency checks, security checks, test suites, and CI commands are aligned with the current architecture, directory structure, and real project risks. Enforce strong typing without bypasses: no unnecessary `any`, casts, ignores, unchecked dynamic access, untyped public APIs, or type-check suppression unless justified and documented.

Implement design patterns only where they simplify or stabilize the architecture, such as Facade, Strategy, Factory, Proxy, Adapter, State, or Observer. Avoid adding patterns unless they reduce coupling, duplication, or instability.

Following the existing pattern, from now on separate what is core from what should be an extension/plugin, including packages that should be installed with `uv pip install package[extra]`.

<Feature>

<Feature/>

