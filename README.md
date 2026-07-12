# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/pdum/dsl/\_\_init\_\_.py        |        2 |        0 |    100% |           |
| src/pdum/dsl/combinators.py         |      162 |       10 |     94% |134, 155, 171, 190, 206-207, 211, 239, 265, 274 |
| src/pdum/dsl/kernel/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/pdum/dsl/kernel/api.py          |        8 |        0 |    100% |           |
| src/pdum/dsl/kernel/cache.py        |      145 |        6 |     96% |107, 125-128, 221 |
| src/pdum/dsl/kernel/capture.py      |       72 |        1 |     99% |       111 |
| src/pdum/dsl/kernel/ir.py           |      107 |        3 |     97% |74, 166, 177 |
| src/pdum/dsl/kernel/lower.py        |       91 |        6 |     93% |73-74, 124, 134, 150, 157 |
| src/pdum/dsl/kernel/ops.py          |       63 |        5 |     92% |77, 93, 118-119, 123 |
| src/pdum/dsl/kernel/pack.py         |      136 |        7 |     95% |143, 148, 202, 258-260, 294 |
| src/pdum/dsl/kernel/printer.py      |       45 |        9 |     80% |33-34, 36, 49-54 |
| src/pdum/dsl/kernel/rewrite.py      |      117 |        5 |     96% |79, 143, 172, 184-185 |
| src/pdum/dsl/kernel/types.py        |       94 |        6 |     94% |88, 97-98, 140, 155, 171 |
| src/pdum/dsl/kernel/valuekind.py    |       80 |        1 |     99% |        85 |
| src/pdum/dsl/stdlib/\_\_init\_\_.py |        0 |        0 |    100% |           |
| src/pdum/dsl/stdlib/base\_lang.py   |       69 |       17 |     75% |34-36, 41, 48, 55, 61, 68, 73-75, 80, 83, 98, 108-109, 112 |
| src/pdum/dsl/viz.py                 |      152 |       17 |     89% |114, 122-123, 154, 175-176, 182-187, 213, 236, 262-268 |
| **TOTAL**                           | **1344** |   **93** | **93%** |           |


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