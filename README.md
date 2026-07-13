# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                             |    Stmts |     Miss |   Cover |   Missing |
|------------------------------------------------- | -------: | -------: | ------: | --------: |
| src/pdum/dsl/\_\_init\_\_.py                     |        6 |        0 |    100% |           |
| src/pdum/dsl/backends/\_emit.py                  |       68 |        1 |     99% |        97 |
| src/pdum/dsl/backends/c.py                       |      163 |       19 |     88% |76, 82, 102, 110, 116, 118, 123, 130-132, 134, 140, 145, 150, 155, 193, 210, 217, 239 |
| src/pdum/dsl/bench.py                            |      112 |       33 |     71% |114-120, 175-201 |
| src/pdum/dsl/combinators.py                      |      162 |        7 |     96% |134, 155, 171, 190, 239, 265, 274 |
| src/pdum/dsl/demo/\_\_init\_\_.py                |        1 |        0 |    100% |           |
| src/pdum/dsl/demo/graphics.py                    |       31 |        9 |     71% |31, 35, 39, 46, 50, 54-55, 65, 68 |
| src/pdum/dsl/demo/simple\_shader/\_\_init\_\_.py |        1 |        0 |    100% |           |
| src/pdum/dsl/demo/simple\_shader/python.py       |       98 |        7 |     93% |57, 66, 70, 88, 100-102 |
| src/pdum/dsl/demo/simple\_shader/wgsl.py         |      244 |      133 |     45% |54, 66-72, 118, 122-123, 126, 128, 132, 134, 138, 142-148, 181-183, 200-209, 218-248, 253-265, 268-290, 296-321, 332-367, 370-402, 411, 416-420 |
| src/pdum/dsl/events.py                           |      153 |       14 |     91% |48, 52-54, 57, 92, 170, 173, 189-196 |
| src/pdum/dsl/kernel/\_\_init\_\_.py              |        1 |        0 |    100% |           |
| src/pdum/dsl/kernel/api.py                       |        5 |        0 |    100% |           |
| src/pdum/dsl/kernel/cache.py                     |      163 |        7 |     96% |66-67, 122-125, 249 |
| src/pdum/dsl/kernel/capture.py                   |       78 |        1 |     99% |       122 |
| src/pdum/dsl/kernel/events.py                    |       41 |        0 |    100% |           |
| src/pdum/dsl/kernel/ir.py                        |      107 |        3 |     97% |74, 164, 175 |
| src/pdum/dsl/kernel/lower.py                     |       91 |        7 |     92% |73-74, 102, 124, 134, 150, 157 |
| src/pdum/dsl/kernel/ops.py                       |       65 |        3 |     95% |81, 97, 127 |
| src/pdum/dsl/kernel/pack.py                      |      136 |        2 |     99% |  141, 289 |
| src/pdum/dsl/kernel/printer.py                   |       45 |        9 |     80% |33-34, 36, 49-54 |
| src/pdum/dsl/kernel/registry.py                  |      117 |        1 |     99% |       167 |
| src/pdum/dsl/kernel/rewrite.py                   |      117 |        2 |     98% |   79, 143 |
| src/pdum/dsl/kernel/types.py                     |       96 |        5 |     95% |99-100, 142, 157, 173 |
| src/pdum/dsl/kernel/valuekind.py                 |       80 |        1 |     99% |        85 |
| src/pdum/dsl/stdlib/\_\_init\_\_.py              |       25 |        1 |     96% |        25 |
| src/pdum/dsl/stdlib/arrays.py                    |      196 |       23 |     88% |50, 73, 85, 95, 98, 114-115, 126, 133, 140-141, 153, 160-161, 213, 222, 238, 286, 297, 338-339, 345-346 |
| src/pdum/dsl/stdlib/base\_lang.py                |      236 |       46 |     81% |40-42, 47, 54, 59, 66-73, 78, 85, 91, 98, 105, 109, 116-124, 128-131, 141, 158, 166, 207, 256, 294, 302, 317-318, 333, 368, 378, 380, 394 |
| src/pdum/dsl/stdlib/batteries.py                 |       38 |        8 |     79% |37, 43, 64, 68, 72, 76-77, 81 |
| src/pdum/dsl/stdlib/surfaces.py                  |       64 |        3 |     95% |88, 104, 112 |
| src/pdum/dsl/stdlib/transforms.py                |      312 |       44 |     86% |86-87, 100-101, 103-109, 111-115, 117-120, 124, 128, 130-131, 136, 143, 146, 178-180, 279, 312, 320, 342, 359, 361, 371, 387, 392, 405, 409, 415-417 |
| src/pdum/dsl/viz.py                              |      166 |       17 |     90% |115, 123-124, 155, 176-177, 183-188, 214, 237, 271-277 |
| **TOTAL**                                        | **3218** |  **406** | **87%** |           |


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