# Repository Coverage

[Full report](https://htmlpreview.github.io/?https://github.com/habemus-papadum/pdum_dsl/blob/python-coverage-comment-action-data/htmlcov/index.html)

| Name                                               |    Stmts |     Miss |   Cover |   Missing |
|--------------------------------------------------- | -------: | -------: | ------: | --------: |
| packages/dsl/src/pdum/dsl/\_\_init\_\_.py          |       32 |        1 |     97% |        33 |
| packages/dsl/src/pdum/dsl/api.py                   |        5 |        0 |    100% |           |
| packages/dsl/src/pdum/dsl/cache.py                 |      169 |        9 |     95% |66-67, 111-112, 133-136, 260 |
| packages/dsl/src/pdum/dsl/capture.py               |       78 |        1 |     99% |       122 |
| packages/dsl/src/pdum/dsl/derivative.py            |      165 |       13 |     92% |113, 192, 194, 214-215, 235, 238, 248, 255, 285, 288, 305, 321 |
| packages/dsl/src/pdum/dsl/derived.py               |       26 |        2 |     92% |     40-42 |
| packages/dsl/src/pdum/dsl/events.py                |       41 |        0 |    100% |           |
| packages/dsl/src/pdum/dsl/intrinsics.py            |       45 |        8 |     82% |41, 47, 71, 75, 79, 83-84, 88 |
| packages/dsl/src/pdum/dsl/ir.py                    |      107 |        3 |     97% |74, 164, 175 |
| packages/dsl/src/pdum/dsl/lower.py                 |      100 |        4 |     96% |73-74, 126, 198 |
| packages/dsl/src/pdum/dsl/markers.py               |       55 |        0 |    100% |           |
| packages/dsl/src/pdum/dsl/naming.py                |       22 |        1 |     95% |        34 |
| packages/dsl/src/pdum/dsl/ops.py                   |       68 |        4 |     94% |54, 87, 103, 133 |
| packages/dsl/src/pdum/dsl/pack.py                  |      138 |        5 |     96% |141, 146, 198, 285, 295 |
| packages/dsl/src/pdum/dsl/pipe.py                  |      133 |        7 |     95% |89, 110, 126, 147, 194, 209, 218 |
| packages/dsl/src/pdum/dsl/printer.py               |       45 |        9 |     80% |33-34, 36, 49-54 |
| packages/dsl/src/pdum/dsl/recorder.py              |      145 |        7 |     95% |48, 52-54, 57, 92, 170, 173 |
| packages/dsl/src/pdum/dsl/reference.py             |       99 |        7 |     93% |66, 76, 80, 91, 103-105 |
| packages/dsl/src/pdum/dsl/registry.py              |      136 |        2 |     99% |  128, 191 |
| packages/dsl/src/pdum/dsl/render.py                |       68 |        1 |     99% |        97 |
| packages/dsl/src/pdum/dsl/rewrite.py               |      117 |        2 |     98% |   79, 143 |
| packages/dsl/src/pdum/dsl/staging.py               |        7 |        0 |    100% |           |
| packages/dsl/src/pdum/dsl/surfaces.py              |       72 |        3 |     96% |89-90, 119 |
| packages/dsl/src/pdum/dsl/types.py                 |      109 |        7 |     94% |89-90, 99-100, 176, 191, 207 |
| packages/dsl/src/pdum/dsl/value.py                 |      251 |       54 |     78% |41-43, 48, 51-58, 60, 67-74, 79, 86, 92, 99, 106, 110, 117-125, 129-132, 142, 159, 169, 210, 259, 289, 297, 312-313, 328, 363, 371, 373, 389, 397, 399 |
| packages/dsl/src/pdum/dsl/valuekind.py             |       80 |        1 |     99% |        85 |
| packages/tensorlib/src/pdum/tl/\_\_init\_\_.py     |       19 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/assemblage.py       |      170 |       16 |     91% |82-83, 95, 97, 99, 112, 161, 167, 209, 214, 222, 228, 231-232, 235, 256 |
| packages/tensorlib/src/pdum/tl/autodiff.py         |      608 |       36 |     94% |123, 203, 229, 258, 283, 424, 482-498, 618, 625, 646, 659-661, 689, 715, 722, 760, 785, 807, 870, 884, 967, 969, 971 |
| packages/tensorlib/src/pdum/tl/buffer.py           |       67 |       13 |     81% |40, 42, 55, 60, 67, 73, 82-83, 102, 107, 112, 117, 122 |
| packages/tensorlib/src/pdum/tl/chart.py            |       74 |        4 |     95% |72, 100, 138, 145 |
| packages/tensorlib/src/pdum/tl/compute.py          |      202 |       16 |     92% |92, 102, 104, 106, 178, 189, 192, 196, 202, 220, 247, 295, 331, 333, 336, 338 |
| packages/tensorlib/src/pdum/tl/coords.py           |      280 |       14 |     95% |171, 248, 251, 406, 410, 415, 417, 419, 424-427, 437, 447 |
| packages/tensorlib/src/pdum/tl/derivative.py       |        3 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/dialect.py          |      786 |      109 |     86% |132, 145, 152, 154, 167, 170, 176-177, 187-194, 200, 212, 216, 218, 220, 253, 311, 330, 348-349, 389-390, 493, 499-500, 535-536, 541, 564, 597, 600, 608, 617, 636, 653, 656-661, 668, 670, 691, 711, 722, 729-736, 742, 745-747, 757-759, 779, 781, 784-786, 842, 846, 850, 857, 895, 985, 989, 1013, 1031, 1034, 1073, 1077, 1096, 1109-1134, 1141, 1156, 1231, 1242, 1245, 1248 |
| packages/tensorlib/src/pdum/tl/dtypes.py           |       19 |        3 |     84% |15, 21, 45 |
| packages/tensorlib/src/pdum/tl/encoding.py         |       93 |        9 |     90% |32, 35, 38, 129, 132, 135-138 |
| packages/tensorlib/src/pdum/tl/graphics.py         |      467 |       89 |     81% |74, 86, 98-109, 136, 154, 168, 195-197, 220-236, 289, 291, 315, 330, 401, 422, 425, 458, 504, 507, 522, 528, 530, 552, 606-607, 613-614, 632, 668, 671, 674, 705, 708, 711, 725-726, 732-733, 741-764, 771-784, 788 |
| packages/tensorlib/src/pdum/tl/guarded.py          |      308 |       46 |     85% |110, 120, 131, 134, 151-152, 164, 170-172, 178, 189-192, 196, 199-200, 211, 217, 220, 226, 233, 248, 257, 260, 297, 309-310, 325-326, 346-347, 370-371, 386, 401-402, 414, 418, 449, 459, 487, 492, 496, 517 |
| packages/tensorlib/src/pdum/tl/indexing.py         |      158 |        2 |     99% |   90, 115 |
| packages/tensorlib/src/pdum/tl/ir.py               |      312 |       20 |     94% |131-134, 137, 169, 215, 265, 280-281, 291, 310, 313, 316, 407, 409, 507, 545-546, 559 |
| packages/tensorlib/src/pdum/tl/kernel.py           |      715 |       62 |     91% |174, 244, 255, 261, 269, 283, 334, 362, 375, 426, 451-453, 470, 477, 485-486, 518, 535, 547, 549, 606, 611, 635, 675-676, 688, 693-694, 699, 703-704, 764, 769, 777, 809-810, 831-834, 877, 882, 886, 888-889, 919, 933, 939, 967, 973, 994, 1003, 1005, 1008-1009, 1047-1049, 1148, 1161, 1171 |
| packages/tensorlib/src/pdum/tl/layout.py           |      549 |       64 |     88% |70-71, 82, 84, 100, 139, 146, 149, 157, 162, 167, 169, 173-180, 190-194, 196-199, 208, 214, 223, 228, 259, 385-388, 399-400, 419, 425, 441, 463, 475, 478, 482, 508, 531, 540, 564, 572, 586, 636, 697, 710, 712, 717, 721, 739, 751, 784, 804, 807, 836, 841 |
| packages/tensorlib/src/pdum/tl/licenses.py         |       22 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/lifting.py          |       57 |        6 |     89% |71, 79-81, 108, 119, 128 |
| packages/tensorlib/src/pdum/tl/markers.py          |       18 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/mdsl.py             |      106 |        7 |     93% |84, 94, 99, 167, 245, 255, 259 |
| packages/tensorlib/src/pdum/tl/memory.py           |      100 |        1 |     99% |        88 |
| packages/tensorlib/src/pdum/tl/nodes.py            |        3 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/opcount.py          |       95 |        2 |     98% |  111, 150 |
| packages/tensorlib/src/pdum/tl/placement.py        |      126 |       14 |     89% |61, 182, 184-192, 203-206 |
| packages/tensorlib/src/pdum/tl/producer.py         |      205 |       52 |     75% |43, 71, 80-81, 94-106, 119-120, 140, 145, 150-152, 154-159, 166, 172, 177, 183, 189-191, 198, 200, 204-207, 215, 221, 229, 232-234, 244, 246, 255, 268, 273, 276, 286, 288 |
| packages/tensorlib/src/pdum/tl/provisioning.py     |       66 |        3 |     95% |67, 71, 76 |
| packages/tensorlib/src/pdum/tl/random.py           |       60 |        1 |     98% |        74 |
| packages/tensorlib/src/pdum/tl/registry.py         |       29 |        2 |     93% |    42, 46 |
| packages/tensorlib/src/pdum/tl/scope.py            |       59 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/signatures.py       |      152 |       12 |     92% |66, 83, 130, 141, 160, 162, 173, 179, 207, 220-221, 244 |
| packages/tensorlib/src/pdum/tl/tensor.py           |      346 |       33 |     90% |69, 119, 126, 154, 170, 180, 250, 274, 304, 310, 316-318, 324, 327, 330, 333, 336, 403, 464, 467-472, 522, 524, 568, 571, 573, 595-597, 603 |
| packages/tensorlib/src/pdum/tl/transforms.py       |      158 |        3 |     98% |141, 186, 193 |
| packages/tensorlib/src/pdum/tl/units.py            |      272 |       37 |     86% |43-51, 94, 103, 115, 127, 130, 163, 190, 192, 196, 202, 211, 221, 224, 231, 256-257, 260-261, 282, 305, 311, 331, 336, 343, 349, 355, 363, 366 |
| packages/tensorlib/src/pdum/tl/zoo/\_\_init\_\_.py |       10 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/zoo/attention.py    |      117 |       27 |     77% |38, 45-47, 51, 94-99, 122-125, 150-153, 178-179, 183-188 |
| packages/tensorlib/src/pdum/tl/zoo/cylinder.py     |       55 |       15 |     73% |49-51, 60-67, 75, 90-92 |
| packages/tensorlib/src/pdum/tl/zoo/gemm.py         |       30 |        4 |     87% |     43-46 |
| packages/tensorlib/src/pdum/tl/zoo/gpt2.py         |      113 |       24 |     79% |48-51, 55-58, 75-83, 96-99, 115, 121-122 |
| packages/tensorlib/src/pdum/tl/zoo/llama.py        |       97 |       21 |     78% |42-45, 64-81 |
| packages/tensorlib/src/pdum/tl/zoo/megatron.py     |       81 |       14 |     83% |     61-74 |
| packages/tensorlib/src/pdum/tl/zoo/moe.py          |       73 |       21 |     71% |     50-78 |
| packages/tensorlib/src/pdum/tl/zoo/physics.py      |       65 |       14 |     78% |31-34, 38-42, 89-96 |
| packages/tensorlib/src/pdum/tl/zoo/trainer.py      |      112 |       41 |     63% |61-63, 66, 70-92, 96-115 |
| packages/tensorlib/src/pdum/tl/zoo/zoo\_common.py  |       53 |        7 |     87% |64-66, 80-81, 85-86 |
| **TOTAL**                                          | **9823** | **1015** | **90%** |           |


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