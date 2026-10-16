import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Alert, App, Button, Empty, Form, Input, InputNumber, Select, Switch } from "antd";

import { useTranslation } from "react-i18next";
import PageShell from "../../../layouts/PageShell";
import { CardSkeleton } from "../../../components/Skeleton";
import PeerOnlyRemoteAlert from "../../../components/PeerOnlyRemoteAlert";
import { acpApi } from "../../../api/modules/acp";
import {
  ACP_DEFAULT_STDIO_BUFFER_LIMIT_BYTES,
  type ACPPassthroughConfig,
  type ACPRunnerConfig,
} from "../../../api/types/acp";
import { useAgent } from "../../../context/AgentContext";
import { blocksAcpOutboundFromConfig } from "../../Experts/components/agentBackendForm";
import { ACPCard } from "./components/ACPCard";
import {
  ACPDrawer,
  formValuesToRunner,
  runnerToFormValues,
} from "./components/ACPDrawer";
import { BUILTIN_RUNNER_ORDER, isBuiltinRunner } from "./constants";
import channelStyles from "../Channels/index.module.less";
import styles from "./index.module.less";

const EMPTY_RUNNERS: Record<string, ACPRunnerConfig> = {};

interface ACPPanelProps {
  /** When set (Experts tools drawer), use this agent instead of the chat selection. */
  agentId?: string | null;
}

/** ACP runners manager — shared by `/acp` and Personalization → Tools. */
export function ACPPanel({ agentId: agentIdProp }: ACPPanelProps = {}) {
  const { t } = useTranslation();
  const { modal, message } = App.useApp();
  const { activeAgentId: contextAgentId, agents } = useAgent();
  const activeAgentId =
    agentIdProp !== undefined ? agentIdProp : contextAgentId;
  /** Only the Experts tools drawer should tunnel account-global ACP runners. */
  const runnersScopeId = agentIdProp;
  const [runners, setRunners] =
    useState<Record<string, ACPRunnerConfig>>(EMPTY_RUNNERS);
  const [toolEnabled, setToolEnabled] = useState(false);
  const [runnersLoading, setRunnersLoading] = useState(true);
  const [toolLoading, setToolLoading] = useState(false);
  const [toolToggleLoading, setToolToggleLoading] = useState(false);
  const [hoverKey, setHoverKey] = useState<string | null>(null);
  const [toggleLoadingKey, setToggleLoadingKey] = useState<string | null>(null);
  const [activeKey, setActiveKey] = useState<string | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [isCreateMode, setIsCreateMode] = useState(false);
  const [saving, setSaving] = useState(false);
  const [form] = Form.useForm();
  const [passthroughEnabled, setPassthroughEnabled] = useState(false);
  const [passthroughLoading, setPassthroughLoading] = useState(false);
  const [passthroughSaving, setPassthroughSaving] = useState(false);
  const [passthroughForm] = Form.useForm();
  const activeAgentIdRef = useRef(activeAgentId);

  const activeAgent = useMemo(
    () => agents.find((row) => row.agent_id === activeAgentId) ?? null,
    [agents, activeAgentId],
  );
  const outboundBlocked = blocksAcpOutboundFromConfig(
    activeAgent?.config ?? null,
  );
  // Under directory sandbox: lock runner enable/edit only. The per-agent
  // acp_runner tool toggle stays available (host-spawned runners may already
  // be enabled for other agents on this account).

  useEffect(() => {
    activeAgentIdRef.current = activeAgentId;
  }, [activeAgentId]);

  useEffect(() => {
    let cancelled = false;
    setRunnersLoading(true);
    void (async () => {
      try {
        const data = await acpApi.getGlobalRunners(runnersScopeId);
        if (!cancelled) {
          setRunners(data.runners || EMPTY_RUNNERS);
        }
      } catch {
        if (!cancelled) {
          message.error(t("acp.loadFailed"));
        }
      } finally {
        if (!cancelled) {
          setRunnersLoading(false);
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [t, runnersScopeId]);

  useEffect(() => {
    if (!activeAgentId) {
      setToolEnabled(false);
      setToolLoading(false);
      return;
    }

    const agentId = activeAgentId;
    setToolLoading(true);

    let cancelled = false;
    void (async () => {
      try {
        const data = await acpApi.getConfig(agentId);
        if (!cancelled && activeAgentIdRef.current === agentId) {
          setToolEnabled(data.tool_enabled);
        }
      } catch {
        if (!cancelled && activeAgentIdRef.current === agentId) {
          message.error(t("acp.loadFailed"));
        }
      } finally {
        if (!cancelled && activeAgentIdRef.current === agentId) {
          setToolLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeAgentId, t]);

  useEffect(() => {
    if (!activeAgentId) {
      setPassthroughEnabled(false);
      setPassthroughLoading(false);
      passthroughForm.resetFields();
      return;
    }

    const agentId = activeAgentId;
    setPassthroughLoading(true);

    let cancelled = false;
    void (async () => {
      try {
        const data = await acpApi.getPassthrough(agentId);
        if (!cancelled && activeAgentIdRef.current === agentId) {
          const cfg = data.acp_passthrough;
          setPassthroughEnabled(Boolean(cfg));
          passthroughForm.setFieldsValue({
            runner: cfg?.runner || undefined,
            cwd: cfg?.cwd || undefined,
            max_runtime: cfg?.max_runtime ?? undefined,
            start_prompt: cfg?.start_prompt || undefined,
          });
        }
      } catch {
        if (!cancelled && activeAgentIdRef.current === agentId) {
          message.error(t("acp.loadFailed"));
        }
      } finally {
        if (!cancelled && activeAgentIdRef.current === agentId) {
          setPassthroughLoading(false);
        }
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [activeAgentId, t, passthroughForm]);

  const orderedKeys = useMemo(() => {
    const keys = Object.keys(runners || {});
    return [
      ...BUILTIN_RUNNER_ORDER.filter((k) => keys.includes(k)),
      ...keys.filter((k) => !isBuiltinRunner(k)).sort(),
    ];
  }, [runners]);

  const enabledRunnerKeys = useMemo(
    () => orderedKeys.filter((k) => runners[k]?.enabled),
    [orderedKeys, runners],
  );

  const cards = useMemo(() => {
    const enabled: { key: string; cfg: ACPRunnerConfig }[] = [];
    const disabled: { key: string; cfg: ACPRunnerConfig }[] = [];
    for (const key of orderedKeys) {
      const cfg = runners[key];
      if (!cfg) continue;
      (cfg.enabled ? enabled : disabled).push({ key, cfg });
    }
    return [...enabled, ...disabled];
  }, [runners, orderedKeys]);

  const persistRunners = useCallback(
    async (next: Record<string, ACPRunnerConfig>) => {
      const saved = await acpApi.updateGlobalRunners(next, runnersScopeId);
      setRunners(saved.runners || EMPTY_RUNNERS);
      return saved.runners;
    },
    [runnersScopeId],
  );

  const persistToolEnabled = useCallback(
    async (agentId: string, checked: boolean) => {
      const saved = await acpApi.updateToolEnabled(agentId, checked);
      if (activeAgentIdRef.current === agentId) {
        setToolEnabled(saved.tool_enabled);
      }
      return saved.tool_enabled;
    },
    [],
  );

  const handleToolToggle = async (checked: boolean) => {
    const agentId = activeAgentIdRef.current;
    if (!agentId || toolLoading) return;
    setToolToggleLoading(true);
    try {
      await persistToolEnabled(agentId, checked);
      message.success(t("acp.configSaved"));
    } catch {
      message.error(t("acp.configFailed"));
    } finally {
      setToolToggleLoading(false);
    }
  };

  const handlePassthroughToggle = (checked: boolean) => {
    setPassthroughEnabled(checked);
  };

  const handlePassthroughSave = async () => {
    const agentId = activeAgentIdRef.current;
    if (!agentId || passthroughLoading) return;
    let values: Record<string, unknown>;
    try {
      values = await passthroughForm.validateFields();
    } catch {
      return;
    }
    const body: ACPPassthroughConfig = {
      enabled: passthroughEnabled,
      runner: String(values.runner ?? ""),
      cwd: values.cwd ? String(values.cwd) : null,
      max_runtime:
        values.max_runtime === undefined || values.max_runtime === null
          ? null
          : Number(values.max_runtime),
      start_prompt: values.start_prompt ? String(values.start_prompt) : null,
    };
    setPassthroughSaving(true);
    try {
      const data = await acpApi.updatePassthrough(agentId, body);
      setPassthroughEnabled(Boolean(data.acp_passthrough));
      message.success(t("acp.passthroughSaved"));
    } catch {
      message.error(t("acp.passthroughSaveFailed"));
    } finally {
      setPassthroughSaving(false);
    }
  };

  const handleToggleEnabled = async (runnerKey: string, checked: boolean) => {
    // Runners are account-global; only block *enabling* under sandbox.
    if (runnersLoading || (outboundBlocked && checked)) return;
    const runner = runners[runnerKey];
    if (!runner) return;
    if (!runner.command?.trim() && checked) {
      message.warning(t("acp.commandRequired"));
      return;
    }
    setToggleLoadingKey(runnerKey);
    try {
      const nextRunners = {
        ...runners,
        [runnerKey]: { ...runner, enabled: checked },
      };
      await persistRunners(nextRunners);
      message.success(t("acp.configSaved"));
    } catch {
      message.error(t("acp.configFailed"));
    } finally {
      setToggleLoadingKey(null);
    }
  };

  const openEdit = (key: string) => {
    if (outboundBlocked) return;
    const cfg = runners[key];
    setIsCreateMode(false);
    setActiveKey(key);
    setDrawerOpen(true);
    form.setFieldsValue(runnerToFormValues(key, cfg));
  };

  const openCreate = () => {
    setIsCreateMode(true);
    setActiveKey(null);
    setDrawerOpen(true);
    form.resetFields();
    form.setFieldsValue({
      runnerKey: "",
      enabled: true,
      command: "",
      argsText: "",
      envText: "",
      trusted: true,
      tool_parse_mode: "update_detail",
      stdio_buffer_limit_bytes: ACP_DEFAULT_STDIO_BUFFER_LIMIT_BYTES,
    });
  };

  const handleSubmit = async (values: Record<string, unknown>) => {
    const targetKey = String(values.runnerKey || activeKey || "").trim();
    if (!targetKey) return;
    if ((isCreateMode || targetKey !== activeKey) && runners[targetKey]) {
      message.error(t("acp.runnerKeyExists"));
      return;
    }
    const runner = formValuesToRunner(values);
    setSaving(true);
    try {
      const nextRunners = { ...runners };
      if (!isCreateMode && activeKey && activeKey !== targetKey) {
        delete nextRunners[activeKey];
      }
      nextRunners[targetKey] = runner;
      await persistRunners(nextRunners);
      setDrawerOpen(false);
      message.success(
        isCreateMode ? t("acp.createSuccess") : t("acp.configSaved"),
      );
    } catch {
      message.error(t("acp.configFailed"));
    } finally {
      setSaving(false);
    }
  };

  const handleDelete = () => {
    if (!activeKey || isBuiltinRunner(activeKey)) return;
    modal.confirm({
      title: t("acp.deleteTitle", { name: activeKey }),
      content: t("acp.deleteConfirm"),
      okText: t("common.delete"),
      cancelText: t("common.cancel"),
      okButtonProps: { danger: true },
      onOk: async () => {
        const nextRunners = { ...runners };
        delete nextRunners[activeKey];
        await persistRunners(nextRunners);
        setDrawerOpen(false);
        message.success(t("acp.deleteSuccess"));
      },
    });
  };

  const toolToggleDisabled = !activeAgentId || toolLoading;

  const toolSwitch = (
    <Switch
      key={activeAgentId ?? "none"}
      checked={toolEnabled}
      loading={toolLoading || toolToggleLoading}
      disabled={toolToggleDisabled}
      onChange={handleToolToggle}
    />
  );

  return (
    <>
      {agentIdProp === undefined ? (
        <PeerOnlyRemoteAlert hintKey="peerOnlyAcp" />
      ) : null}
      <div className={styles.toolbar}>
        <div className={styles.toolbarText}>
          <div className={styles.description}>{t("acp.description")}</div>
          <p className={styles.scopeHint}>{t("acp.globalRunnersHint")}</p>
        </div>
        <Button type="primary" onClick={openCreate}>
          {t("acp.create")}
        </Button>
      </div>

      {activeAgentId && outboundBlocked ? (
        <Alert
          type="info"
          showIcon
          className={styles.outboundBlockedAlert}
          message={t("acp.outboundBlockedHint")}
        />
      ) : null}

      {runnersLoading && cards.length === 0 ? (
        <CardSkeleton count={4} />
      ) : (
        <div className={channelStyles.channelsGrid}>
          {cards.map(({ key, cfg }) => (
            <ACPCard
              key={key}
              runnerKey={key}
              config={cfg}
              isHover={hoverKey === key}
              toggleLoading={toggleLoadingKey === key}
              interactionDisabled={outboundBlocked}
              onClick={() => openEdit(key)}
              onMouseEnter={() => setHoverKey(key)}
              onMouseLeave={() => setHoverKey(null)}
              onToggleEnabled={handleToggleEnabled}
            />
          ))}
        </div>
      )}

      <div className={styles.agentToolSection}>
        <p className={styles.scopeHint}>{t("acp.toolEnabledHint")}</p>
        {!activeAgentId ? (
          <Empty
            description={t("mbtiPage.pickAgent")}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <div className={styles.toolToggle}>
            <span>{t("acp.toolEnabled")}</span>
            {toolSwitch}
          </div>
        )}
      </div>

      <div className={styles.passthroughSection}>
        <div className={styles.toolToggle}>
          <span>{t("acp.passthroughTitle")}</span>
          <Switch
            checked={passthroughEnabled}
            loading={passthroughLoading}
            disabled={!activeAgentId || passthroughLoading}
            onChange={handlePassthroughToggle}
          />
        </div>
        {!activeAgentId ? (
          <Empty
            description={t("mbtiPage.pickAgent")}
            image={Empty.PRESENTED_IMAGE_SIMPLE}
          />
        ) : (
          <>
            <p className={styles.scopeHint}>{t("acp.passthroughHint")}</p>
            {passthroughEnabled && enabledRunnerKeys.length === 0 ? (
              <Alert
                type="warning"
                showIcon
                className={styles.passthroughAlert}
                message={t("acp.passthroughNoRunnerHint")}
              />
            ) : null}
            {passthroughEnabled ? (
              <Form
                form={passthroughForm}
                layout="vertical"
                className={styles.passthroughForm}
              >
                <Form.Item
                  label={t("acp.passthroughRunnerLabel")}
                  name="runner"
                  rules={[
                    {
                      required: true,
                      message: t("acp.passthroughRunnerRequired"),
                    },
                  ]}
                >
                  <Select
                    placeholder={t("acp.passthroughRunnerPlaceholder")}
                    options={enabledRunnerKeys.map((key) => ({
                      value: key,
                      label: key,
                    }))}
                  />
                </Form.Item>
                <Form.Item
                  label={t("acp.passthroughCwdLabel")}
                  name="cwd"
                >
                  <Input placeholder={t("acp.passthroughCwdPlaceholder")} />
                </Form.Item>
                <Form.Item
                  label={t("acp.passthroughMaxRuntimeLabel")}
                  name="max_runtime"
                >
                  <InputNumber
                    min={1}
                    placeholder="300"
                    style={{ width: "100%" }}
                  />
                </Form.Item>
                <Form.Item
                  label={t("acp.passthroughStartPromptLabel")}
                  name="start_prompt"
                >
                  <Input
                    placeholder={t("acp.passthroughStartPromptPlaceholder")}
                  />
                </Form.Item>
              </Form>
            ) : null}
            <Button
              type="primary"
              loading={passthroughSaving}
              disabled={passthroughLoading}
              onClick={handlePassthroughSave}
            >
              {t("common.save")}
            </Button>
          </>
        )}
      </div>

      <ACPDrawer
        open={drawerOpen}
        activeKey={activeKey}
        isCreateMode={isCreateMode}
        form={form}
        saving={saving}
        canEditKey={
          isCreateMode || (activeKey ? !isBuiltinRunner(activeKey) : false)
        }
        canDelete={
          !isCreateMode && activeKey !== null && !isBuiltinRunner(activeKey)
        }
        onClose={() => setDrawerOpen(false)}
        onSubmit={handleSubmit}
        onDelete={handleDelete}
      />
    </>
  );
}

/** Standalone ACP page (sidebar Control entry). */
export default function ACPPage() {
  const { t } = useTranslation();
  return (
    <PageShell
      title={t("pageShell.acp.title")}
      subtitle={t("pageShell.acp.subtitle")}
      agentScoped
    >
      <ACPPanel />
    </PageShell>
  );
}
