# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/pdum/dsl/\_\_init\_\_.py                     |        6 |        0 |    100% |           |
| src/pdum/dsl/backends/\_emit.py                  |       56 |        3 |     95% |     59-61 |
| src/pdum/dsl/combinators.py                      |      162 |        7 |     96% |134, 155, 171, 190, 239, 265, 274 |
| src/pdum/dsl/demo/\_\_init\_\_.py                |        1 |        0 |    100% |           |
| src/pdum/dsl/demo/simple\_shader/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/pdum/dsl/demo/simple\_shader/python.py       |       71 |       15 |     79% |57, 64, 69, 72-83 |
| src/pdum/dsl/demo/simple\_shader/wgsl.py         |      207 |      100 |     52% |53, 65-71, 117, 121-122, 125, 127, 131, 133, 137, 141, 174-175, 192-201, 210-240, 243-276, 287-322, 325-357, 366, 371-375 |
| src/pdum/dsl/kernel/\_\_init\_\_.py              |        1 |        0 |    100% |           |
| src/pdum/dsl/kernel/api.py                       |        7 |        0 |    100% |           |
| src/pdum/dsl/kernel/cache.py                     |      158 |        8 |     95% |68-69, 110, 128-131, 237 |
| src/pdum/dsl/kernel/capture.py                   |       75 |        1 |     99% |       122 |
| src/pdum/dsl/kernel/ir.py                        |      107 |        3 |     97% |74, 166, 177 |
| src/pdum/dsl/kernel/lower.py                     |       91 |        6 |     93% |73-74, 124, 134, 150, 157 |
| src/pdum/dsl/kernel/ops.py                       |       63 |        5 |     92% |77, 93, 118-119, 123 |
| src/pdum/dsl/kernel/pack.py                      |      136 |        7 |     95% |143, 148, 202, 258-260, 294 |
| src/pdum/dsl/kernel/printer.py                   |       45 |        9 |     80% |33-34, 36, 49-54 |
| src/pdum/dsl/kernel/registry.py                  |       76 |        2 |     97% |   80, 141 |
| src/pdum/dsl/kernel/rewrite.py                   |      117 |        2 |     98% |   79, 143 |
| src/pdum/dsl/kernel/types.py                     |       94 |        6 |     94% |88, 97-98, 140, 155, 171 |
| src/pdum/dsl/kernel/valuekind.py                 |       80 |        1 |     99% |        85 |
| src/pdum/dsl/stdlib/\_\_init\_\_.py              |       13 |        0 |    100% |           |
| src/pdum/dsl/stdlib/base\_lang.py                |       69 |       17 |     75% |34-36, 41, 48, 55, 61, 68, 73-75, 80, 83, 98, 108-109, 112 |
| src/pdum/dsl/viz.py                              |      152 |       17 |     89% |114, 122-123, 154, 175-176, 182-187, 213, 236, 262-268 |
| **TOTAL**                                        | **1788** |  **209** | **88%** |           |


## Setup coverage badge

Below are examples of the badges you can use in your main branch `README` file.

### Direct image

[![Coverage badge](https://raw.githubusercontent.com/habemus-papadum/pdum_dsl/python-coverage-comment-action-data/badge.svg)](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

This is the one to use if your repository is private or if you don't want to customize anything.

### [Shields.io](https://shields.io) Json Endpoint

[![Coverage badge](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/habemus-papadum/pdum_dsl/python-coverage-comment-action-data/endpoint.json)](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

Using this one will allow you to [customize](https://shields.io/endpoint) the look of your badge.
It won't work with private repositories. It won't be refreshed more than once per five minutes.

### [Shields.io](https://shields.io) Dynamic Badge

[![Coverage badge](https://img.shields.io/badge/dynamic/json?color=brightgreen&label=coverage&query=%24.message&url=https%3A%2F%2Fraw.githubusercontent.com%2Fhabemus-papadum%2Fpdum_dsl%2Fpython-coverage-comment-action-data%2Fendpoint.json)](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

This one will always be the same color. It won't work for private repos. I'm not even sure why we included it.

## What is that?

This branch is part of the
[python-coverage-comment-action](https://github.com/marketplace/actions/python-coverage-comment)
GitHub Action. All the files in this branch are automatically generated and may be
overwritten at any moment.