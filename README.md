# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------ | -------: | -------: | ------: | --------: |
| src/pdum/dsl/\_\_init\_\_.py        |        2 |        0 |    100% |           |
| src/pdum/dsl/combinators.py         |      137 |        7 |     95% |134, 155, 171, 190, 202-203, 231 |
| src/pdum/dsl/kernel/\_\_init\_\_.py |        0 |        0 |    100% |           |
| src/pdum/dsl/kernel/api.py          |        8 |        0 |    100% |           |
| src/pdum/dsl/kernel/cache.py        |      145 |        6 |     96% |107, 125-128, 221 |
| src/pdum/dsl/kernel/capture.py      |       68 |        1 |     99% |       101 |
| src/pdum/dsl/kernel/ir.py           |      107 |        4 |     96% |74, 166, 177, 181 |
| src/pdum/dsl/kernel/ops.py          |       61 |        5 |     92% |77, 91, 116-117, 121 |
| src/pdum/dsl/kernel/printer.py      |       45 |        9 |     80% |33-34, 36, 49-54 |
| src/pdum/dsl/kernel/rewrite.py      |      114 |        5 |     96% |73, 137, 166, 176-177 |
| src/pdum/dsl/kernel/types.py        |       94 |        6 |     94% |88, 97-98, 140, 155, 171 |
| src/pdum/dsl/kernel/valuekind.py    |       56 |        0 |    100% |           |
| **TOTAL**                           |  **837** |   **43** | **95%** |           |


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