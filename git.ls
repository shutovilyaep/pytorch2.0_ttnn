Reb to:
    commit 08f67677958bddb5a109554dfd056eb46ca76801
    Author: Jonathan Baker <jbaker@tenstorrent.com>
    Date:   Thu Sep 4 12:10:16 2025 -0400


+        -[Local] commit 07ae4531e25faf4af7dd566692e387156533a4bb -> build/v0.60.1
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Wed Oct 15 14:54:45 2025 +0000

            script upd

        scripts/venv_recreate.sh | 7 +++++--
        1 file changed, 5 insertions(+), 2 deletions(-)

+        -[Local] commit 64ca8e8462e214117668a49153684201ba4f9643
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Wed Oct 15 14:13:08 2025 +0000

            script upd: venv cleanup

        scripts/venv_recreate.sh | 4 +++-
        1 file changed, 3 insertions(+), 1 deletion(-)


+        -[Local] commit 82b7cdf1e8236a25e5c8de085394faf30820fed8
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Tue Oct 14 13:15:13 2025 +0000

            DEBUG SCRIPT: Remove C++ extension rebuild step from run_cpp_extension_tests.sh

        scripts/run_cpp_extension_tests.sh | 10 +++++-----
        1 file changed, 5 insertions(+), 5 deletions(-)


++[CMake, INC, to squash later?] commit cce73bdb11d30b73407b70fb3f1389e2029d473d
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Tue Oct 14 13:14:43 2025 +0000

    in progress: CMake modifcations to include tt-metal deps

 torch_ttnn/cpp_extension/CMakeLists.txt            |  4 +-
 .../cpp_extension/third-party/CMakeLists.txt       | 53 ----------------------
 2 files changed, 2 insertions(+), 55 deletions(-)


+        -[Local] commit 47610cab924767465f7e436eefc2ce444204c18f
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Tue Oct 14 13:03:33 2025 +0000

            SCRIPT: direct upd, clean

        scripts/direct.sh | 6 +++++-
        1 file changed, 5 insertions(+), 1 deletion(-)


+        -[Local] commit bb9df3ab1b7e1a170b46797b0a01161324526ef2
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Fri Oct 10 16:35:03 2025 +0000

            DEBUG: scripts, status md

        PRIORITIES_STATUS.md                | 115 +++++++++
        scripts/direct.sh                   |  56 +++++
        scripts/gen_ttnn_ops.py             | 477 ++++++++++++++++++++++++++++++++++++
        scripts/rewrite_compile_commands.py |  49 ++++
        scripts/venv_recreate.sh            |  10 +-
        5 files changed, 702 insertions(+), 5 deletions(-)


+        -[Local] commit db2065b74f539fbc5c0bada14e70cd6c0f0cff2e
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Fri Oct 10 16:33:53 2025 +0000

            tests run!

        scripts/run_cpp_extension_tests.sh | 104 +++++++++++++++++++++++++++++++++++++
        1 file changed, 104 insertions(+)


++[CMake, INC, to squash later?] commit b700235e2aacfa657e886a4437f3ebc07129623e
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Fri Oct 10 14:27:39 2025 +0000

    in progress, sobmodule tt-metal build

 torch_ttnn/cpp_extension/CMakeLists.txt         | 12 ++++++++++--
 torch_ttnn/cpp_extension/build_cpp_extension.sh | 14 +++++++++++---
 2 files changed, 21 insertions(+), 5 deletions(-)


+        -[Local] commit 52e2ea5182991335b0a253528de522ce84bf5ec8
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Fri Sep 19 14:56:30 2025 +0000

            tags

        scripts/tags | 3 +--
        1 file changed, 1 insertion(+), 2 deletions(-)

+        -[Local] commit 1d0c807c27a03ed0af4afe8f3cdb4755de63030f -> build/v0.60.0
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Fri Sep 19 14:48:05 2025 +0000

            tags

        scripts/tags | 7 +++++--
        1 file changed, 5 insertions(+), 2 deletions(-)

+        -[Local] commit 2062a8f4ab0796894b26ebc89e6ca3773bf05971 -> build/v0.59.1.upd
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Fri Sep 19 14:33:33 2025 +0000

            tags

        scripts/tags | 9 +++++++--
        1 file changed, 7 insertions(+), 2 deletions(-)


++[INC] commit 556cab0db52c16c119b78758c0814acecc9b1415
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Fri Sep 19 14:32:06 2025 +0000

    Build fix: Update namespace from 'tt::stl' to 'ttsl' for SmallVector and Span usage in copy and tensor implementation files

 .../cpp_extension/ttnn_cpp_extension/src/core/TtnnTensorImpl.cpp      | 2 +-
 torch_ttnn/cpp_extension/ttnn_cpp_extension/src/core/copy.cpp         | 4 ++--
 torch_ttnn/cpp_extension/ttnn_cpp_extension/src/ops/creation.cpp      | 2 +-
 3 files changed, 4 insertions(+), 4 deletions(-)



++[INC, TO SQUASH LATER, commented in a script] commit 00d208b8c5d60b10c0edc03f8a6659205467d12b
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Fri Sep 19 14:30:50 2025 +0000

    scripts upd - not to rebuild tt-metal in build_cpp_extension

 torch_ttnn/cpp_extension/build_cpp_extension.sh | 20 ++++++++++----------
 1 file changed, 10 insertions(+), 10 deletions(-)



++[copy_data removed, tt-metal API change revert, TO SQUASH LATER?] commit d3dd08317399e33e1611979c7c737f3e9a7677f9
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Fri Sep 19 14:30:27 2025 +0000

    tt-metal API change fix

 .../include/ttnn_cpp_extension/core/TtnnCustomAllocator.hpp        | 4 +---
 .../ttnn_cpp_extension/src/core/TtnnCustomAllocator.cpp            | 7 +------
 2 files changed, 2 insertions(+), 9 deletions(-)



++[build scripts are updated ~] commit 396a1586fc95e3964e3b23b5d39e2e84e5659a4d
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Fri Sep 19 14:26:54 2025 +0000

    build fix attempt, C++ errors are left

 torch_ttnn/cpp_extension/build_cpp_extension.sh | 10 ++++++++--
 1 file changed, 8 insertions(+), 2 deletions(-)



+    -[Split, build scripts are updated ~] commit 5a88f3da03c64b58df3a68236de3d4cbc543b845
    Author: Ilia Shutov <Ilia_Shutov@epam.com>
    Date:   Fri Sep 19 14:07:20 2025 +0000

        SCRITPS+DEBUG: scripts upd: large build_cpp_extension - to use the same toolchain as tt-metal

    scripts/build_cpp_direct.sh                     |  7 ++--
    scripts/venv_recreate.sh                        |  4 ++
    torch_ttnn/cpp_extension/build_cpp_extension.sh | 56 +++++++++++--------------
    torch_ttnn/cpp_extension/setup.py               |  8 +++-
    4 files changed, 40 insertions(+), 35 deletions(-)


+    -[Split, build scripts are updated ~] commit 3d0e6e878037ea1b4b921e8c9b1d1f5d7dea8391
    Author: Ilia Shutov <Ilia_Shutov@epam.com>
    Date:   Thu Sep 18 13:54:45 2025 +0000

        Build fix attempt

    scripts/build_cpp_direct.sh                     | 2 +-
    torch_ttnn/cpp_extension/build_cpp_extension.sh | 2 +-
    torch_ttnn/cpp_extension/pyproject.toml         | 2 +-
    3 files changed, 3 insertions(+), 3 deletions(-)


++[copy_data added, tt-metal API change INC, TO SQUASH LATER?] commit c19458dee193e88bb1519e535cc2cac9a9ae2d0f
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Thu Sep 18 13:48:16 2025 +0000

    Build fix: updated TtnnCustomAllocator; added copy_data method for compatibility with updated API

 .../include/ttnn_cpp_extension/core/TtnnCustomAllocator.hpp    |  9 ++++++---
 .../ttnn_cpp_extension/src/core/TtnnCustomAllocator.cpp        | 10 ++++++++--
 2 files changed, 14 insertions(+), 5 deletions(-)


++[PyTorch INC, TO SQUASH LATER, fastfix, will be changed?] commit ea697f696601dbf70d1e557dbb6a190b109debb3
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Thu Sep 18 13:43:11 2025 +0000

    Build attempt, setup.py fix

 torch_ttnn/cpp_extension/setup.py | 22 +++++++++++++++++++---
 1 file changed, 19 insertions(+), 3 deletions(-)


++[INC] commit 160de3ff7b15fa0142b1d7edcc254684ddd4fbf6
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Thu Sep 18 11:41:48 2025 +0000

    Build fix: to_layout API changed

 torch_ttnn/cpp_extension/ttnn_cpp_extension/src/ops/binary.cpp | 4 ++--
 1 file changed, 2 insertions(+), 2 deletions(-)


~[build/v0.59.1 is not set in commit message]        -[Local] commit 5d0741962a1ddb0246cee3202e0008f45e3ba793 -> build/v0.59.1
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Thu Sep 18 13:05:05 2025 +0000

            tags

        scripts/tags | 10 +++++++---
        1 file changed, 7 insertions(+), 3 deletions(-)

+        -[Local] commit afc2a244a1b734b05aa8d384e7930f94169a4a2e
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Thu Sep 18 12:40:15 2025 +0000

            tags

        scripts/tags | 12 +++++++-----
        1 file changed, 7 insertions(+), 5 deletions(-)

+        -[Local] commit c51df6bd1c3eb70a443bbb79ddd84e19865b0728
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Thu Sep 18 11:41:31 2025 +0000

            tags

        scripts/tags | 14 +++++++-------
        1 file changed, 7 insertions(+), 7 deletions(-)


+        -[Local] commit c0b879fd1aadb2350f00c1bd3a2cfa1191c1c607 -> build/v0.59.0-rc10 .. build/v0.59.0-rc50
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Mon Sep 15 14:45:36 2025 +0000

            DEBUG: script iterating tt-metal versions

        scripts/build_cpp_extension_artifacts.sh | 119 ++++++++++++++++++++++++++-----
        1 file changed, 103 insertions(+), 16 deletions(-)


+        -[Local] commit 50d7999b3516aa778bfc682dcf2627811c39a1a0
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Mon Sep 15 14:45:15 2025 +0000

            Build fix

        torch_ttnn/cpp_extension/utils/get_torch_abi_flags.py | 7 ++++---
        1 file changed, 4 insertions(+), 3 deletions(-)


[~build/v0.59.1 added] ++[CMake INC, TO SQUASH LATER, fastfix, will be changed] commit 93266e1aa7635f13a54b062fe01aeed528437040
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Mon Sep 15 13:58:49 2025 +0000

    Build fix

torch_ttnn/cpp_extension/CMakeLists.txt | 1 +
1 file changed, 1 insertion(+)


+        -[Local] commit 03917d82003f4b30edd6b1afd34e6ad776fc19d4
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Fri Sep 12 11:05:05 2025 +0000

            Migration notes for v0.59.0: Implemented build fixes for C++ extension following tt-metal submodule updates, including adjustments to include directories, handling of deprecated APIs, and resolution of third-party include errors. Ensured successful compilation and addressed deprecation warnings.

        scripts/v0.59.0.fix | 51 +++++++++++++++++++++++++++++++++++++++++++++++++++
        1 file changed, 51 insertions(+)


++[INC] commit 0b869a466f7d20bfa3ba2ad6d703c71eb54d860e
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Fri Sep 12 11:01:18 2025 +0000

    Refactor: Replace instances of ttnn::SmallVector with tt::stl::SmallVector for logical shape handling in TtnnTensorImpl, creation operations, and vector utilities.

 .../cpp_extension/ttnn_cpp_extension/src/core/TtnnTensorImpl.cpp      | 4 ++--
 torch_ttnn/cpp_extension/ttnn_cpp_extension/src/ops/creation.cpp      | 4 ++--
 .../cpp_extension/ttnn_cpp_extension/src/utils/vector_utils.cpp       | 2 +-
 3 files changed, 5 insertions(+), 5 deletions(-)


++[CMake INC, TO SQUASH LATER, fastfix, will be changed]
commit 1d57348923435aaa82a567ff8696fb9a48535ce7
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Fri Sep 12 10:59:02 2025 +0000

    Build fix: CMake configuration, include additional directories for TTNN dependencies - tt-logger and spdlog

torch_ttnn/cpp_extension/cmake/find_prebuilt_ttnn.cmake | 5 ++++-
1 file changed, 4 insertions(+), 1 deletion(-)


+        -[Local] commit caae39634cf70403d8c47f8a69d67665be672742
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Fri Sep 12 10:48:47 2025 +0000

            DEBUG: scripts, tags info

        scripts/build_cpp_extension_artifacts.sh | 119 +-----
        scripts/tags                             | 685 +++++++++++++++++++++++++++++++
        2 files changed, 701 insertions(+), 103 deletions(-)


+        -[Local] commit 7684ca75c95d86cea610cd4de10e378f64a2d870 -> build/v0.58.1
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Wed Sep 17 17:00:54 2025 +0000

        Build fix: install using --use-pep517

        scripts/build_cpp_direct.sh | 5 +++++
        1 file changed, 5 insertions(+)


+        ?-[Local] commit 60398cb83a64759c024667b059cb065088bbdab3
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Wed Sep 17 16:41:25 2025 +0000

            Build fix: MPICC, MPICXX are set

        torch_ttnn/cpp_extension/build_cpp_extension.sh | 5 +++++
        1 file changed, 5 insertions(+)


+        -[Local] commit 6c8669b447a9290c3de71c201d4a37e154bfb4ab
        Author: Ilia Shutov <Ilia_Shutov@epam.com>
        Date:   Wed Sep 17 13:19:17 2025 +0000

            DEBUG: scripts

        scripts/build_cpp_direct.sh              |  11 +++
        scripts/build_cpp_extension_artifacts.sh | 161 +++++++++++++++++++++++++++++++
        scripts/metal_checkout.sh                |  10 ++
        scripts/test_script.py                   |  18 ++++
        scripts/venv_recreate.sh                 |  12 +++
        5 files changed, 212 insertions(+)


+    -[Split] commit af7f11d51c394e31e350db9a4ff350b5173a05d1
    Author: Ilia Shutov <Ilia_Shutov@epam.com>
    Date:   Thu Sep 11 14:03:29 2025 +0000

    Update dependencies in requirements-dev.txt and setup.py for compatibility with tt-metal v0.58.1; add migration notes for v0.58.1 API changes in new script.

    requirements-dev.txt |  9 +++++----
    scripts/v0.58.1.fix  | 52 ++++++++++++++++++++++++++++++++++++++++++++++++++++
    setup.py             |  2 +-
    3 files changed, 58 insertions(+), 5 deletions(-)


+ +[INC] commit e5a6fab033859f3d4579a841b07abeb30d911b8f
Author: Ilia Shutov <Ilia_Shutov@epam.com>
Date:   Thu Sep 11 13:52:16 2025 +0000

    v0.58.1 API change: ttnn_copy_from fix to use borrowed HostBuffer with lifetime pin

 .../ttnn_cpp_extension/src/core/copy.cpp           | 28 +++++++++++-----------
 1 file changed, 14 insertions(+), 14 deletions(-)
