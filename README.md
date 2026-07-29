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
| packages/dsl/src/pdum/dsl/reference.py             |      102 |        7 |     93% |66, 76, 80, 91, 103-105 |
| packages/dsl/src/pdum/dsl/registry.py              |      136 |        2 |     99% |  128, 191 |
| packages/dsl/src/pdum/dsl/render.py                |       68 |        1 |     99% |        97 |
| packages/dsl/src/pdum/dsl/rewrite.py               |      117 |        2 |     98% |   79, 143 |
| packages/dsl/src/pdum/dsl/staging.py               |        7 |        0 |    100% |           |
| packages/dsl/src/pdum/dsl/surfaces.py              |       72 |        3 |     96% |89-90, 119 |
| packages/dsl/src/pdum/dsl/types.py                 |      109 |        7 |     94% |89-90, 99-100, 176, 191, 207 |
| packages/dsl/src/pdum/dsl/value.py                 |      269 |       55 |     80% |41-43, 48, 51-58, 60, 67-74, 79, 86, 92, 99, 106, 110, 117-125, 129-132, 142, 159, 169, 210, 259, 289, 297, 312-313, 332, 340, 385, 393, 395, 411, 419, 421 |
| packages/dsl/src/pdum/dsl/valuekind.py             |       80 |        1 |     99% |        85 |
| packages/tensorlib/src/pdum/tl/\_\_init\_\_.py     |       19 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/assemblage.py       |      172 |       15 |     91% |83-84, 96, 98, 113, 163, 169, 211, 216, 224, 230, 233-234, 237, 258 |
| packages/tensorlib/src/pdum/tl/autodiff.py         |      664 |       35 |     95% |166, 241, 294, 317, 438, 488-498, 584, 590, 604, 616-618, 641, 667, 675, 708, 717, 741, 762, 824, 837, 878, 917, 919, 921 |
| packages/tensorlib/src/pdum/tl/buffer.py           |       67 |       13 |     81% |40, 42, 55, 60, 67, 73, 82-83, 102, 107, 112, 117, 122 |
| packages/tensorlib/src/pdum/tl/chart.py            |       74 |        4 |     95% |72, 100, 138, 145 |
| packages/tensorlib/src/pdum/tl/compute.py          |      221 |       17 |     92% |92, 102, 104, 106, 178, 189, 192, 196, 202, 220, 247, 295, 331, 333, 336, 338, 361 |
| packages/tensorlib/src/pdum/tl/coords.py           |      280 |       14 |     95% |171, 248, 251, 406, 410, 415, 417, 419, 424-427, 437, 447 |
| packages/tensorlib/src/pdum/tl/derivative.py       |        3 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/dialect.py          |      929 |      114 |     88% |143, 156, 163, 165, 178, 181, 187-188, 198-205, 211, 230, 232, 234, 238, 243, 245, 256, 264, 269, 313, 410-411, 511, 530, 548-549, 589-590, 693, 699-700, 735-736, 741, 764, 797, 800, 808, 817, 836, 853, 856-861, 868, 870, 891, 911, 922, 929-936, 942, 945-947, 957-959, 979, 981, 984-986, 1042, 1046, 1050, 1057, 1167-1168, 1197, 1287, 1291, 1315, 1333, 1336, 1375, 1379, 1398, 1411-1436, 1443, 1471 |
| packages/tensorlib/src/pdum/tl/dtypes.py           |       19 |        3 |     84% |15, 21, 45 |
| packages/tensorlib/src/pdum/tl/encoding.py         |       93 |        9 |     90% |32, 35, 38, 129, 132, 135-138 |
| packages/tensorlib/src/pdum/tl/graphics.py         |      450 |       87 |     81% |74, 86, 98-109, 136, 169-171, 192-208, 261, 263, 287, 302, 373, 394, 397, 430, 476, 479, 494, 500, 502, 525, 579-580, 586-587, 605, 641, 644, 647, 678, 681, 684, 698-699, 705-706, 714-737, 744-757, 761 |
| packages/tensorlib/src/pdum/tl/guarded.py          |      308 |       46 |     85% |110, 120, 131, 134, 151-152, 164, 170-172, 178, 189-192, 196, 199-200, 211, 217, 220, 226, 233, 248, 257, 260, 297, 309-310, 325-326, 346-347, 370-371, 386, 401-402, 414, 418, 449, 459, 487, 492, 496, 517 |
| packages/tensorlib/src/pdum/tl/indexing.py         |      158 |        2 |     99% |   90, 115 |
| packages/tensorlib/src/pdum/tl/kernel.py           |      817 |       72 |     91% |181, 254, 265, 271, 279, 293, 344, 372, 385, 436, 461-463, 480, 487, 495-496, 528, 545, 557, 559, 687, 690, 716, 765-766, 778, 783-784, 789, 793-794, 859, 867, 899-900, 921-924, 967, 972, 976, 978-979, 1009, 1023, 1029, 1057, 1063, 1084, 1093, 1095, 1098-1099, 1137-1139, 1150, 1172-1174, 1176, 1193, 1208-1211, 1224, 1326, 1339, 1349 |
| packages/tensorlib/src/pdum/tl/layout.py           |      557 |       63 |     89% |70-71, 82, 84, 100, 139, 146, 149, 157, 162, 167, 169, 173-180, 190-194, 196-199, 208, 214, 223, 228, 259, 385-388, 399-400, 419, 425, 441, 463, 475, 478, 482, 508, 531, 540, 564, 572, 586, 636, 697, 710, 712, 717, 721, 751, 784, 804, 807, 836, 841 |
| packages/tensorlib/src/pdum/tl/licenses.py         |       23 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/lifting.py          |       59 |        6 |     90% |70, 78-80, 105, 116, 125 |
| packages/tensorlib/src/pdum/tl/markers.py          |       33 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/mdsl.py             |      106 |        7 |     93% |84, 94, 99, 167, 245, 255, 259 |
| packages/tensorlib/src/pdum/tl/memory.py           |      119 |        3 |     97% |96, 113, 136 |
| packages/tensorlib/src/pdum/tl/nodes.py            |        3 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/opcount.py          |      117 |        3 |     97% |108, 122, 170 |
| packages/tensorlib/src/pdum/tl/placement.py        |      140 |       12 |     91% |59, 130, 145, 210, 214-217, 223, 229-232 |
| packages/tensorlib/src/pdum/tl/producer.py         |      205 |       52 |     75% |43, 71, 80-81, 94-106, 119-120, 140, 145, 150-152, 154-159, 166, 172, 177, 183, 189-191, 198, 200, 204-207, 215, 221, 229, 232-234, 244, 246, 255, 268, 273, 276, 286, 288 |
| packages/tensorlib/src/pdum/tl/provisioning.py     |       66 |        3 |     95% |67, 71, 76 |
| packages/tensorlib/src/pdum/tl/random.py           |       60 |        1 |     98% |        74 |
| packages/tensorlib/src/pdum/tl/registry.py         |       29 |        2 |     93% |    42, 46 |
| packages/tensorlib/src/pdum/tl/scope.py            |       58 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/signatures.py       |      174 |       13 |     93% |66, 83, 130, 141, 160, 162, 173, 209, 231, 242, 254-255, 285 |
| packages/tensorlib/src/pdum/tl/tensor.py           |      372 |       36 |     90% |69, 119, 126, 154, 170, 180, 280, 304, 334, 340, 346-348, 354, 357, 360, 363, 366, 433, 494, 497-502, 552, 554, 598, 601, 603, 625-627, 633, 662, 665, 668 |
| packages/tensorlib/src/pdum/tl/transforms.py       |      228 |       10 |     96% |101, 207, 220, 309, 335-340 |
| packages/tensorlib/src/pdum/tl/units.py            |      272 |       37 |     86% |43-51, 94, 103, 115, 127, 130, 163, 190, 192, 196, 202, 211, 221, 224, 231, 256-257, 260-261, 282, 305, 311, 331, 336, 343, 349, 355, 363, 366 |
| packages/tensorlib/src/pdum/tl/zoo/\_\_init\_\_.py |       10 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/zoo/attention.py    |      116 |       27 |     77% |37, 44-46, 50, 93-98, 121-124, 149-152, 177-178, 182-187 |
| packages/tensorlib/src/pdum/tl/zoo/cylinder.py     |       55 |       15 |     73% |50-52, 61-68, 76, 91-93 |
| packages/tensorlib/src/pdum/tl/zoo/gemm.py         |       30 |        4 |     87% |     43-46 |
| packages/tensorlib/src/pdum/tl/zoo/gpt2.py         |      112 |       24 |     79% |47-50, 54-57, 74-82, 95-98, 114, 120-121 |
| packages/tensorlib/src/pdum/tl/zoo/llama.py        |       96 |       21 |     78% |41-44, 63-80 |
| packages/tensorlib/src/pdum/tl/zoo/megatron.py     |       80 |       14 |     82% |     60-73 |
| packages/tensorlib/src/pdum/tl/zoo/moe.py          |       73 |       21 |     71% |     50-78 |
| packages/tensorlib/src/pdum/tl/zoo/physics.py      |       65 |       14 |     78% |32-35, 39-43, 90-97 |
| packages/tensorlib/src/pdum/tl/zoo/tiles.py        |      144 |        0 |    100% |           |
| packages/tensorlib/src/pdum/tl/zoo/trainer.py      |      112 |       41 |     63% |61-63, 66, 70-92, 96-115 |
| packages/tensorlib/src/pdum/tl/zoo/zoo\_common.py  |       53 |        7 |     87% |64-66, 80-81, 85-86 |
| **TOTAL**                                          | **10175** | **1019** | **90%** |           |


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