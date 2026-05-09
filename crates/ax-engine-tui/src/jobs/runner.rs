use crate::jobs::plan::JobSpec;
use std::io::Read;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::{Duration, Instant};
use thiserror::Error;

const DEFAULT_LOG_TAIL_LINES: usize = 80;

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum JobStatus {
    Succeeded,
    Failed,
    Canceled,
}

impl JobStatus {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Succeeded => "succeeded",
            Self::Failed => "failed",
            Self::Canceled => "canceled",
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct JobOutput {
    pub job_id: String,
    pub status: JobStatus,
    pub exit_code: Option<i32>,
    pub log_tail: Vec<String>,
}

pub struct RunningJob {
    spec: JobSpec,
    child: Child,
    log_tail_lines: usize,
    started_at: Instant,
}

impl RunningJob {
    pub fn start(spec: JobSpec) -> Result<Self, JobRunnerError> {
        let mut command = Command::new(&spec.command.program);
        command.args(&spec.command.args);
        if let Some(cwd) = spec.command.cwd.as_deref() {
            command.current_dir(cwd);
        }
        let child = command
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|source| JobRunnerError::Spawn {
                job_id: spec.id.clone(),
                source,
            })?;
        Ok(Self {
            spec,
            child,
            log_tail_lines: DEFAULT_LOG_TAIL_LINES,
            started_at: Instant::now(),
        })
    }

    pub fn wait(mut self) -> Result<JobOutput, JobRunnerError> {
        let status = self.child.wait().map_err(|source| JobRunnerError::Wait {
            job_id: self.spec.id.clone(),
            source,
        })?;
        self.finish(status)
    }

    pub fn cancel(mut self) -> Result<JobOutput, JobRunnerError> {
        if self
            .child
            .try_wait()
            .map_err(|source| JobRunnerError::Wait {
                job_id: self.spec.id.clone(),
                source,
            })?
            .is_none()
        {
            self.child.kill().map_err(|source| JobRunnerError::Cancel {
                job_id: self.spec.id.clone(),
                source,
            })?;
        }
        let status = self.child.wait().map_err(|source| JobRunnerError::Wait {
            job_id: self.spec.id.clone(),
            source,
        })?;
        let mut output = self.finish(status)?;
        output.status = JobStatus::Canceled;
        Ok(output)
    }

    pub fn wait_for_startup(&mut self, timeout: Duration) -> Result<bool, JobRunnerError> {
        let deadline = Instant::now() + timeout;
        loop {
            if self
                .child
                .try_wait()
                .map_err(|source| JobRunnerError::Wait {
                    job_id: self.spec.id.clone(),
                    source,
                })?
                .is_some()
            {
                return Ok(false);
            }
            if Instant::now() >= deadline {
                return Ok(true);
            }
            std::thread::sleep(Duration::from_millis(10));
        }
    }

    pub fn started_at(&self) -> Instant {
        self.started_at
    }

    fn finish(&mut self, status: ExitStatus) -> Result<JobOutput, JobRunnerError> {
        let mut logs = Vec::new();
        read_pipe_tail(self.child.stdout.take(), "stdout", &mut logs)?;
        read_pipe_tail(self.child.stderr.take(), "stderr", &mut logs)?;
        Ok(JobOutput {
            job_id: self.spec.id.clone(),
            status: if status.success() {
                JobStatus::Succeeded
            } else {
                JobStatus::Failed
            },
            exit_code: status.code(),
            log_tail: trim_tail(logs, self.log_tail_lines),
        })
    }
}

fn read_pipe_tail(
    pipe: Option<impl Read>,
    stream: &str,
    logs: &mut Vec<String>,
) -> Result<(), JobRunnerError> {
    let Some(mut pipe) = pipe else {
        return Ok(());
    };
    let mut text = String::new();
    pipe.read_to_string(&mut text)
        .map_err(JobRunnerError::ReadLog)?;
    logs.extend(text.lines().map(|line| format!("{stream}: {line}")));
    Ok(())
}

fn trim_tail(mut logs: Vec<String>, max_lines: usize) -> Vec<String> {
    if logs.len() <= max_lines {
        return logs;
    }
    logs.drain(0..logs.len() - max_lines);
    logs
}

pub fn run_to_completion(spec: JobSpec) -> Result<JobOutput, JobRunnerError> {
    RunningJob::start(spec)?.wait()
}

#[derive(Debug, Error)]
pub enum JobRunnerError {
    #[error("failed to spawn job {job_id}: {source}")]
    Spawn {
        job_id: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to wait for job {job_id}: {source}")]
    Wait {
        job_id: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to cancel job {job_id}: {source}")]
    Cancel {
        job_id: String,
        #[source]
        source: std::io::Error,
    },
    #[error("failed to read job logs: {0}")]
    ReadLog(std::io::Error),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::jobs::plan::{CommandInvocation, EvidenceClass, JobKind, JobSpec};

    fn shell_job(id: &str, script: &str) -> JobSpec {
        JobSpec::new(
            id,
            id,
            JobKind::ServerSmoke,
            EvidenceClass::RouteContract,
            CommandInvocation::new("sh", vec!["-c".to_string(), script.to_string()], None),
        )
    }

    fn sleep_job(id: &str, seconds: &str) -> JobSpec {
        JobSpec::new(
            id,
            id,
            JobKind::ServerLaunch,
            EvidenceClass::RouteContract,
            CommandInvocation::new("sleep", vec![seconds.to_string()], None),
        )
    }

    #[test]
    fn captures_log_tail_for_completed_job() {
        let output = run_to_completion(shell_job("fake-smoke", "printf 'ready\\n'"))
            .expect("fake job should run");

        assert_eq!(output.status, JobStatus::Succeeded);
        assert_eq!(output.log_tail, vec!["stdout: ready"]);
    }

    #[test]
    fn cancel_cleans_up_owned_child_process() {
        let mut job =
            RunningJob::start(sleep_job("fake-server", "30")).expect("fake server should start");

        assert!(
            job.wait_for_startup(Duration::from_millis(50))
                .expect("startup wait should work")
        );
        let output = job.cancel().expect("cancel should clean up process");

        assert_eq!(output.status, JobStatus::Canceled);
        assert_eq!(output.job_id, "fake-server");
    }
}
