from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ClusterRuntimeContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        package_root = Path(os.environ["CTIR_PACKAGE_ROOT"]).resolve()
        cls.code_root = Path(os.environ["CTIR_CODE_ROOT"]).resolve()
        cls.overlay = (
            package_root / "repo_overlay"
            if (package_root / "repo_overlay").is_dir()
            else package_root
        )
        cls.scripts = cls.overlay / "scripts"
        cls.workflow = cls.scripts / "CTIR-MULTI-T1-T5"

    @staticmethod
    def _write_executable(path: Path, body: str) -> None:
        path.write_text(body, encoding="utf-8")
        path.chmod(0o755)

    @staticmethod
    def _argument_value(arguments: list[str], flag: str) -> str:
        index = arguments.index(flag)
        return arguments[index + 1]

    def test_profile_is_loaded_before_nounset(self):
        submit = (self.scripts / "submit_ctir_multitask_h100x_4gpu.sh").read_text()
        self.assertLess(submit.index("source /etc/profile"), submit.index("set -u"))
        self.assertIn("set -eo pipefail", submit[:submit.index("source /etc/profile")])
        for script in (
            self.scripts / "slurm_srun_4gpu.sh",
            self.scripts / "slurm_rank_worker.sh",
            self.workflow / "launch_slurm.sh",
            self.workflow / "run_cl_slurm.sh",
            self.workflow / "train_stage_slurm.sh",
            self.workflow / "eval_stage_slurm.sh",
        ):
            self.assertNotIn("/etc/profile", script.read_text(), script)

    def test_runtime_pythonpath_imports_both_package_styles(self):
        env = os.environ.copy()
        env["PYTHONPATH"] = f"{self.code_root / 'src'}:{self.code_root}"
        prepare = self.workflow / "prepare_multitask_probes.py"
        subprocess.run(
            [sys.executable, str(prepare), "--help"],
            cwd=self.code_root,
            env=env,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import src.ctir.multitask_probe_dataset; import train.train_utils",
            ],
            cwd=self.code_root,
            env=env,
            check=True,
        )

    def test_srun_layout_is_four_one_gpu_ranks_across_two_nodes(self):
        with tempfile.TemporaryDirectory(prefix="ctir-slurm-layout-") as raw_tmp:
            tmp = Path(raw_tmp)
            fake_bin = tmp / "bin"
            fake_bin.mkdir()
            capture = tmp / "srun.args"
            self._write_executable(
                fake_bin / "scontrol",
                "#!/bin/sh\nprintf '%s\\n' hn35 hn39\n",
            )
            self._write_executable(
                fake_bin / "srun",
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CTIR_CAPTURE\"\n",
            )
            env = os.environ.copy()
            env.update({
                "PATH": f"{fake_bin}:{env['PATH']}",
                "CTIR_CAPTURE": str(capture),
                "SLURM_JOB_ID": "241079",
                "SLURM_JOB_NODELIST": "hn[35,39]",
                "MRCL_WORLD_SIZE": "4",
                "MRCL_CPUS_PER_TASK": "14",
            })
            subprocess.run(
                ["/bin/bash", str(self.scripts / "slurm_srun_4gpu.sh"), "/fake/python", "train.py"],
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            arguments = capture.read_text().splitlines()
            self.assertIn("--nodes=2", arguments)
            self.assertIn("--ntasks=4", arguments)
            self.assertIn("--cpus-per-task=14", arguments)
            self.assertIn("--gpus-per-task=1", arguments)
            self.assertIn("--gpu-bind=single:1", arguments)
            self.assertIn(str(self.scripts / "slurm_rank_worker.sh"), arguments)

    def test_rank_worker_exports_train_and_eval_identity(self):
        for mode in ("train", "eval"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory(prefix="ctir-rank-worker-") as raw_tmp:
                tmp = Path(raw_tmp)
                capture = tmp / "python.env"
                fake_python = tmp / "python"
                self._write_executable(
                    fake_python,
                    "#!/bin/sh\n"
                    "if [ \"${1:-}\" = -c ]; then exit 0; fi\n"
                    "{\n"
                    "  printf 'RANK=%s\\n' \"${RANK-unset}\"\n"
                    "  printf 'WORLD_SIZE=%s\\n' \"${WORLD_SIZE-unset}\"\n"
                    "  printf 'LOCAL_RANK=%s\\n' \"${LOCAL_RANK-unset}\"\n"
                    "  printf 'MRCL_SHARD_RANK=%s\\n' \"${MRCL_SHARD_RANK-unset}\"\n"
                    "  printf 'MRCL_NUM_SHARDS=%s\\n' \"${MRCL_NUM_SHARDS-unset}\"\n"
                    "  printf 'ARGS=%s\\n' \"$*\"\n"
                    "} > \"$CTIR_CAPTURE\"\n",
                )
                env = os.environ.copy()
                env.update({
                    "CTIR_CAPTURE": str(capture),
                    "SLURM_JOB_ID": "241082",
                    "SLURM_PROCID": "3",
                    "SLURM_NTASKS": "4",
                    "SLURM_LOCALID": "1",
                    "MASTER_ADDR": "hn35",
                    "MASTER_PORT": "21082",
                    "MRCL_WORKER_MODE": mode,
                })
                subprocess.run(
                    ["/bin/bash", str(self.scripts / "slurm_rank_worker.sh"), str(fake_python), "entry.py", "--x", "1"],
                    env=env,
                    check=True,
                )
                values = dict(line.split("=", 1) for line in capture.read_text().splitlines())
                if mode == "train":
                    self.assertEqual(values["RANK"], "3")
                    self.assertEqual(values["WORLD_SIZE"], "4")
                    self.assertEqual(values["LOCAL_RANK"], "0")
                    self.assertEqual(values["MRCL_SHARD_RANK"], "unset")
                else:
                    self.assertEqual(values["RANK"], "unset")
                    self.assertEqual(values["WORLD_SIZE"], "unset")
                    self.assertEqual(values["LOCAL_RANK"], "unset")
                    self.assertEqual(values["MRCL_SHARD_RANK"], "3")
                    self.assertEqual(values["MRCL_NUM_SHARDS"], "4")
                self.assertEqual(values["ARGS"], "entry.py --x 1")

    def _capture_train_stage(self, mode: str, task_id: int) -> list[str]:
        with tempfile.TemporaryDirectory(prefix="ctir-train-stage-") as raw_tmp:
            tmp = Path(raw_tmp)
            fake_bin = tmp / "bin"
            fake_bin.mkdir()
            capture = tmp / "bash.args"
            self._write_executable(
                fake_bin / "bash",
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CTIR_CAPTURE\"\n",
            )
            env = os.environ.copy()
            env.update({
                "PATH": f"{fake_bin}:{env['PATH']}",
                "CTIR_CAPTURE": str(capture),
                "SLURM_JOB_ID": "241080",
                "TRAIN_PYTHON": "/bundle/envs/trlQwen/bin/python",
                "BASE_MODEL": "/bundle/models/Qwen3-VL-4B-Instruct",
                "BASE_PATH": "/bundle/datasets/MRCL",
                "CTIR_PROBE_ROOT": "/bundle/CPO/experiments/ctir_multitask_t1_t5/probes",
                "MRCL_WORLD_SIZE": "4",
            })
            subprocess.run(
                ["/bin/bash", str(self.workflow / "train_stage_slurm.sh"), mode, str(task_id)],
                cwd=tmp,
                env=env,
                check=True,
                stdout=subprocess.DEVNULL,
            )
            return capture.read_text().splitlines()

    def test_train_stage_matrix_and_preflight_arguments(self):
        t1 = self._capture_train_stage("formal", 1)
        self.assertNotIn("--ctir_multitask_enable", t1)
        self.assertEqual(self._argument_value(t1, "--data_path"), "/bundle/datasets/MRCL/MedBookVQA/jsons/train/data.json")

        t2 = self._capture_train_stage("formal", 2)
        self.assertEqual(
            self._argument_value(t2, "--model_id"),
            "./checkpoints/Qwen3-VL-4B/CTIR-MULTI-CL/training/MedBookVQA",
        )
        self.assertEqual(self._argument_value(t2, "--ctir_multitask_continual_start_step"), "300")
        self.assertEqual(self._argument_value(t2, "--ctir_multitask_probe_index_path").split("/")[-1], "T2.json")

        preflight = self._capture_train_stage("preflight", 5)
        expected = {
            "--model_id": "/bundle/models/Qwen3-VL-4B-Instruct",
            "--data_path": "/bundle/datasets/MRCL/FinMME/jsons/train/data.json",
            "--max_steps": "300",
            "--per_device_train_batch_size": "8",
            "--gradient_accumulation_steps": "4",
            "--ctir_multitask_probe_count": "32",
            "--ctir_multitask_force_beta": "1.0",
            "--ctir_multitask_stop_after_steps": "2",
            "--ctir_multitask_continual_start_step": "1200",
        }
        for flag, value in expected.items():
            self.assertEqual(self._argument_value(preflight, flag), value, flag)
        self.assertEqual(
            self._argument_value(preflight, "--deepspeed"),
            "scripts/zero3_offload_h100_80gb.json",
        )

    def test_formal_stage_refuses_any_nonempty_output_directory(self):
        with tempfile.TemporaryDirectory(prefix="ctir-existing-output-") as raw_tmp:
            tmp = Path(raw_tmp)
            output = tmp / "checkpoints/Qwen3-VL-4B/CTIR-MULTI-CL/training/MedBookVQA"
            output.mkdir(parents=True)
            (output / "partial-artifact.txt").write_text("do not overwrite\n", encoding="utf-8")
            env = os.environ.copy()
            env.update({
                "TRAIN_PYTHON": "/unused/python",
                "BASE_MODEL": "/unused/model",
                "BASE_PATH": "/unused/data",
                "CTIR_PROBE_ROOT": "/unused/probes",
            })
            result = subprocess.run(
                ["/bin/bash", str(self.workflow / "train_stage_slurm.sh"), "formal", "1"],
                cwd=tmp,
                env=env,
                text=True,
                stderr=subprocess.PIPE,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("non-empty formal-stage output directory", result.stderr)

    def test_local_launcher_maps_train_and_eval_to_four_gpus(self):
        with tempfile.TemporaryDirectory(prefix="ctir-local-launcher-") as raw_tmp:
            tmp = Path(raw_tmp)
            capture = tmp / "calls.log"
            fake_python = tmp / "python"
            self._write_executable(
                fake_python,
                "#!/bin/sh\n"
                "printf '%s|%s\\n' \"${CUDA_VISIBLE_DEVICES-all}\" \"$*\" >> \"$CTIR_CAPTURE\"\n",
            )
            env = os.environ.copy()
            env.update({
                "CTIR_CAPTURE": str(capture),
                "CUDA_VISIBLE_DEVICES": "4,5,6,7",
                "MRCL_WORLD_SIZE": "4",
            })
            env["MRCL_WORKER_MODE"] = "train"
            subprocess.run(
                ["/bin/bash", str(self.scripts / "local_4gpu.sh"), str(fake_python), "train.py", "--x", "1"],
                env=env,
                check=True,
            )
            train_call = capture.read_text()
            self.assertIn("-m torch.distributed.run --standalone --nproc-per-node=4 --no-python", train_call)
            self.assertIn(str(self.scripts / "local_rank_worker_4gpu.sh"), train_call)

            capture.write_text("", encoding="utf-8")
            env["MRCL_WORKER_MODE"] = "eval"
            subprocess.run(
                ["/bin/bash", str(self.scripts / "local_4gpu.sh"), str(fake_python), "eval.py"],
                env=env,
                check=True,
            )
            devices = sorted(line.split("|", 1)[0] for line in capture.read_text().splitlines())
            self.assertEqual(devices, ["4", "5", "6", "7"])

    def test_local_rank_worker_exposes_one_physical_gpu(self):
        with tempfile.TemporaryDirectory(prefix="ctir-local-worker-") as raw_tmp:
            tmp = Path(raw_tmp)
            capture = tmp / "worker.env"
            fake_python = tmp / "python"
            self._write_executable(
                fake_python,
                "#!/bin/sh\n"
                "if [ \"${1:-}\" = -c ]; then exit 0; fi\n"
                "printf 'CUDA_VISIBLE_DEVICES=%s\\nLOCAL_RANK=%s\\nRANK=%s\\nWORLD_SIZE=%s\\nMRCL_NODE_LOCAL_RANK=%s\\nARGS=%s\\n' "
                "\"$CUDA_VISIBLE_DEVICES\" \"$LOCAL_RANK\" \"$RANK\" \"$WORLD_SIZE\" "
                "\"$MRCL_NODE_LOCAL_RANK\" \"$*\" > \"$CTIR_CAPTURE\"\n",
            )
            env = os.environ.copy()
            env.update({
                "CTIR_CAPTURE": str(capture),
                "CUDA_VISIBLE_DEVICES": "4,5,6,7",
                "LOCAL_RANK": "2",
                "RANK": "2",
                "WORLD_SIZE": "4",
                "MASTER_ADDR": "localhost",
                "MASTER_PORT": "29500",
                "MRCL_RUN_ID": "worker-test",
            })
            subprocess.run(
                ["/bin/bash", str(self.scripts / "local_rank_worker_4gpu.sh"), str(fake_python), "train.py"],
                env=env,
                check=True,
            )
            values = dict(line.split("=", 1) for line in capture.read_text().splitlines())
            self.assertEqual(values["CUDA_VISIBLE_DEVICES"], "6")
            self.assertEqual(values["LOCAL_RANK"], "0")
            self.assertEqual(values["RANK"], "2")
            self.assertEqual(values["WORLD_SIZE"], "4")
            self.assertEqual(values["MRCL_NODE_LOCAL_RANK"], "2")

    def test_online_launcher_reaches_preflight_and_all_five_stages(self):
        with tempfile.TemporaryDirectory(prefix="ctir-online-chain-") as raw_tmp:
            root = Path(raw_tmp) / "CPO"
            shutil.copytree(self.scripts, root / "scripts")
            (root / "src/dataset").mkdir(parents=True)
            (root / "src/dataset/prompts_2.yaml").write_text("{}\n", encoding="utf-8")
            for entrypoint in ("src/train/train_grpo.py", "src/eval/inference.py", "src/eval/eval.py"):
                path = root / entrypoint
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n", encoding="utf-8")
            model = Path(raw_tmp) / "model"
            model.mkdir()
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            data = Path(raw_tmp) / "data"
            for task in ("MedBookVQA", "Navigation", "We-Math2", "Puzzle", "FinMME"):
                task_root = data / task
                (task_root / "jsons/train").mkdir(parents=True)
                (task_root / "jsons/test").mkdir(parents=True)
                (task_root / "images").mkdir()
                (task_root / "jsons/train/data.json").write_text("[]\n", encoding="utf-8")
                (task_root / "jsons/test/data.json").write_text("[]\n", encoding="utf-8")

            capture = Path(raw_tmp) / "calls.log"
            fake_python = Path(raw_tmp) / "python"
            self._write_executable(
                fake_python,
                "#!/bin/sh\n"
                "printf '%s|%s\\n' \"${CUDA_VISIBLE_DEVICES-all}\" \"$*\" >> \"$CTIR_CAPTURE\"\n",
            )
            env = os.environ.copy()
            env.update({
                "BASE_MODEL": str(model),
                "BASE_PATH": str(data),
                "TRAIN_PYTHON": str(fake_python),
                "EVAL_PYTHON": str(fake_python),
                "CUDA_VISIBLE_DEVICES": "0,1,2,3",
                "CTIR_CAPTURE": str(capture),
                "MRCL_RUN_ID": "online-test",
            })
            result = subprocess.run(
                ["/bin/bash", str(root / "scripts/CTIR-MULTI-T1-T5/launch_online_h100_4gpu.sh")],
                cwd=root,
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            calls = capture.read_text().splitlines()
            torchrun_calls = [line for line in calls if "-m torch.distributed.run" in line]
            inference_calls = [line for line in calls if "src/eval/inference.py --base_model" in line]
            self.assertEqual(len(torchrun_calls), 6)
            self.assertEqual(len(inference_calls), 60)
            self.assertIn("Starting formal T1/5", result.stdout)
            self.assertIn("Starting formal T5/5", result.stdout)
            self.assertIn("EXP-CTIR-MULTI-T1-T5-001 complete", result.stdout)

    def test_eval_stage_uses_four_independent_tp1_shards(self):
        with tempfile.TemporaryDirectory(prefix="ctir-eval-stage-") as raw_tmp:
            tmp = Path(raw_tmp)
            fake_bin = tmp / "bin"
            fake_bin.mkdir()
            bash_capture = tmp / "bash.args"
            python_capture = tmp / "python.args"
            self._write_executable(
                fake_bin / "bash",
                "#!/bin/sh\nprintf '%s\\n' \"$@\" > \"$CTIR_BASH_CAPTURE\"\n",
            )
            self._write_executable(
                fake_bin / "python",
                "#!/bin/sh\nprintf '%s\\n' \"$@\" >> \"$CTIR_PYTHON_CAPTURE\"\n",
            )
            env = os.environ.copy()
            env.update({
                "PATH": f"{fake_bin}:{env['PATH']}",
                "CTIR_BASH_CAPTURE": str(bash_capture),
                "CTIR_PYTHON_CAPTURE": str(python_capture),
                "SLURM_JOB_ID": "241081",
                "EVAL_PYTHON": str(fake_bin / "python"),
                "BASE_PATH": "/bundle/datasets/MRCL",
                "MRCL_WORLD_SIZE": "4",
            })
            subprocess.run(
                ["/bin/bash", str(self.workflow / "eval_stage_slurm.sh"), "1"],
                cwd=tmp,
                env=env,
                check=True,
            )
            arguments = bash_capture.read_text().splitlines()
            self.assertIn("src/eval/inference.py", arguments)
            self.assertEqual(self._argument_value(arguments, "--tensor_parallel_size"), "1")
            self.assertEqual(self._argument_value(arguments, "--batch_size"), "2048")
            self.assertIn("scripts/merge_eval_shards.py", python_capture.read_text())

    def test_relocated_job_body_reaches_all_training_and_eval_stages(self):
        with tempfile.TemporaryDirectory(prefix="ctir-full-chain-") as raw_tmp:
            bundle = Path(raw_tmp) / "mrcl_bundle"
            cpo = bundle / "CPO"
            shutil.copytree(self.scripts, cpo / "scripts")
            (cpo / "scripts/merge_eval_shards.py").write_text("\n", encoding="utf-8")
            (cpo / "src/dataset").mkdir(parents=True)
            (cpo / "src/dataset/prompts_2.yaml").write_text("{}\n", encoding="utf-8")
            for entrypoint in ("src/train/train_grpo.py", "src/eval/inference.py", "src/eval/eval.py"):
                path = cpo / entrypoint
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("\n", encoding="utf-8")
            model = bundle / "models/Qwen3-VL-4B-Instruct"
            model.mkdir(parents=True)
            (model / "config.json").write_text("{}\n", encoding="utf-8")
            for task in ("MedBookVQA", "Navigation", "We-Math2", "Puzzle", "FinMME"):
                task_root = bundle / "datasets/MRCL" / task
                (task_root / "jsons/train").mkdir(parents=True)
                (task_root / "jsons/test").mkdir(parents=True)
                (task_root / "images").mkdir()
                (task_root / "jsons/train/data.json").write_text("[]\n", encoding="utf-8")
                (task_root / "jsons/test/data.json").write_text("[]\n", encoding="utf-8")

            capture = Path(raw_tmp) / "calls.log"
            fake_python_body = (
                "#!/bin/sh\n"
                "printf 'python %s\\n' \"$*\" >> \"$CTIR_CAPTURE\"\n"
            )
            for env_name in ("trlQwen", "vllmQwen"):
                interpreter = bundle / "envs" / env_name / "bin/python"
                interpreter.parent.mkdir(parents=True)
                self._write_executable(interpreter, fake_python_body)

            fake_bin = Path(raw_tmp) / "bin"
            fake_bin.mkdir()
            self._write_executable(fake_bin / "scontrol", "#!/bin/sh\nprintf '%s\\n' hn35 hn39\n")
            self._write_executable(
                fake_bin / "srun",
                "#!/bin/sh\nprintf 'srun %s\\n' \"$*\" >> \"$CTIR_CAPTURE\"\n",
            )
            self._write_executable(fake_bin / "nvcc", "#!/bin/sh\nexit 0\n")
            fake_profile = Path(raw_tmp) / "profile"
            fake_profile.write_text(
                ': "${LC_IDENTIFICATION}"\nmodule() { return 0; }\n',
                encoding="utf-8",
            )
            relocated = Path(raw_tmp) / "tmp/slurmd/job241083/submit.sh"
            relocated.parent.mkdir(parents=True)
            job_body = (self.scripts / "submit_ctir_multitask_h100x_4gpu.sh").read_text()
            job_body = job_body.replace(
                "BUNDLE_ROOT=/XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl/mrcl_bundle",
                f"BUNDLE_ROOT={bundle}",
            ).replace(
                "HOME_ROOT=/XYAIFS00/HOME/sysu_shenli/sysu_shenli_2",
                f"HOME_ROOT={Path(raw_tmp) / 'home'}",
            ).replace(
                "HDD_USER_ROOT=/XYAIFS00/HDD_POOL/sysu_shenli/sysu_shenli_2/cyl",
                f"HDD_USER_ROOT={Path(raw_tmp) / 'hdd'}",
            ).replace(
                "source /etc/profile >/dev/null 2>&1 || true",
                f"source {fake_profile} >/dev/null 2>&1 || true",
            )
            self._write_executable(relocated, job_body)
            env = os.environ.copy()
            env.pop("LC_IDENTIFICATION", None)
            env.update({
                "PATH": f"{fake_bin}:{env['PATH']}",
                "CTIR_CAPTURE": str(capture),
                "SLURM_JOB_ID": "241083",
                "SLURM_JOB_NODELIST": "hn[35,39]",
                "CUDA_HOME": "/APP/u22/ai_x86/CUDA/12.4",
                "MRCL_EXPECTED_GPU": "H100",
            })
            result = subprocess.run(
                ["/bin/bash", str(relocated)],
                cwd=relocated.parent,
                env=env,
                check=True,
                text=True,
                stdout=subprocess.PIPE,
            )
            calls = capture.read_text().splitlines()
            self.assertEqual(sum(line.startswith("srun ") for line in calls), 21)
            self.assertIn("Starting formal T1/5", result.stdout)
            self.assertIn("Starting formal T5/5", result.stdout)
            self.assertIn("EXP-CTIR-MULTI-T1-T5-001 complete", result.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
