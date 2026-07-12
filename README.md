# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/pdum/dsl/\_\_init\_\_.py                     |        6 |        0 |    100% |           |
| src/pdum/dsl/backends/\_emit.py                  |       56 |        3 |     95% |     59-61 |
| src/pdum/dsl/combinators.py                      |      162 |        7 |     96% |134, 155, 171, 190, 239, 265, 274 |
| src/pdum/dsl/demo/\_\_init\_\_.py                |        1 |        0 |    100% |           |
| src/pdum/dsl/demo/simple\_shader/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/pdum/dsl/demo/simple\_shader/python.py       |       79 |       10 |     87% |57, 64, 73-76, 78, 87-89 |
| src/pdum/dsl/demo/simple\_shader/wgsl.py         |      215 |      106 |     51% |54, 66-72, 118, 122-123, 126, 128, 132, 134, 138, 142-148, 181-182, 199-208, 217-247, 250-283, 294-329, 332-364, 373, 378-382 |
| src/pdum/dsl/kernel/\_\_init\_\_.py              |        1 |        0 |    100% |           |
| src/pdum/dsl/kernel/api.py                       |        5 |        0 |    100% |           |
| src/pdum/dsl/kernel/cache.py                     |      158 |        8 |     95% |68-69, 110, 128-131, 237 |
| src/pdum/dsl/kernel/capture.py                   |       75 |        1 |     99% |       121 |
| src/pdum/dsl/kernel/ir.py                        |      107 |        3 |     97% |74, 164, 175 |
| src/pdum/dsl/kernel/lower.py                     |       91 |        6 |     93% |73-74, 124, 134, 150, 157 |
| src/pdum/dsl/kernel/ops.py                       |       65 |        5 |     92% |81, 97, 122-123, 127 |
| src/pdum/dsl/kernel/pack.py                      |      136 |        4 |     97% |141, 146, 198, 289 |
| src/pdum/dsl/kernel/printer.py                   |       45 |        9 |     80% |33-34, 36, 49-54 |
| src/pdum/dsl/kernel/registry.py                  |       91 |        1 |     99% |       137 |
| src/pdum/dsl/kernel/rewrite.py                   |      117 |        2 |     98% |   79, 143 |
| src/pdum/dsl/kernel/types.py                     |       94 |        6 |     94% |88, 97-98, 140, 155, 171 |
| src/pdum/dsl/kernel/valuekind.py                 |       80 |        1 |     99% |        85 |
| src/pdum/dsl/stdlib/\_\_init\_\_.py              |       21 |        1 |     95% |        25 |
| src/pdum/dsl/stdlib/base\_lang.py                |      137 |       47 |     66% |40-42, 47, 50-57, 59, 66-73, 78, 85, 91, 98, 103-105, 109, 116-124, 128-131, 141, 158, 166, 180, 190, 192, 206 |
| src/pdum/dsl/stdlib/batteries.py                 |       55 |       13 |     76% |34, 40, 61, 65, 69, 73-74, 78, 82, 86, 90, 103, 106 |
| src/pdum/dsl/stdlib/surfaces.py                  |       64 |        3 |     95% |88, 104, 112 |
| src/pdum/dsl/viz.py                              |      152 |       17 |     89% |114, 122-123, 154, 175-176, 182-187, 213, 236, 270-276 |
| **TOTAL**                                        | **2014** |  **253** | **87%** |           |


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