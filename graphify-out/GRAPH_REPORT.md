# Graph Report - .  (2026-05-26)

## Corpus Check
- 68 files · ~101,444 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 637 nodes · 1130 edges · 51 communities (30 shown, 21 thin omitted)
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 243 edges (avg confidence: 0.64)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_PyTorch Device and Tensor Utils|PyTorch Device and Tensor Utils]]
- [[_COMMUNITY_Network Evaluation and Baseline Comparison|Network Evaluation and Baseline Comparison]]
- [[_COMMUNITY_SUMO Environment State Utilities|SUMO Environment State Utilities]]
- [[_COMMUNITY_Multi-TLS Environment and Domain Randomization|Multi-TLS Environment and Domain Randomization]]
- [[_COMMUNITY_Double-DQN Agent|Double-DQN Agent]]
- [[_COMMUNITY_WebSocket AI Server|WebSocket AI Server]]
- [[_COMMUNITY_SUMO Junction Graph Nodes|SUMO Junction Graph Nodes]]
- [[_COMMUNITY_FRAP State and Adjacency|FRAP State and Adjacency]]
- [[_COMMUNITY_SUMO Calibration Data Loading|SUMO Calibration Data Loading]]
- [[_COMMUNITY_ATLAS V2 Architecture Components|ATLAS V2 Architecture Components]]
- [[_COMMUNITY_Adaptive Traffic Infrastructure|Adaptive Traffic Infrastructure]]
- [[_COMMUNITY_Academic TSC References|Academic TSC References]]
- [[_COMMUNITY_Regression Gate Tests|Regression Gate Tests]]
- [[_COMMUNITY_Krauss Car-Following Calibration|Krauss Car-Following Calibration]]
- [[_COMMUNITY_Traffic Signal Policy Baselines|Traffic Signal Policy Baselines]]
- [[_COMMUNITY_Training Ablation Logs|Training Ablation Logs]]
- [[_COMMUNITY_Corridor Adjacency Builder|Corridor Adjacency Builder]]
- [[_COMMUNITY_MAPPO Shared Actor Policy|MAPPO Shared Actor Policy]]
- [[_COMMUNITY_SUMO GUI Simulation Launcher|SUMO GUI Simulation Launcher]]
- [[_COMMUNITY_PyTorch Selection Rationale|PyTorch Selection Rationale]]
- [[_COMMUNITY_Krauss Calibration Scripts|Krauss Calibration Scripts]]
- [[_COMMUNITY_3D Viewer WebSockets Frontend|3D Viewer WebSockets Frontend]]
- [[_COMMUNITY_Real-Life Simulation Data Processing|Real-Life Simulation Data Processing]]
- [[_COMMUNITY_3D City Viewer Frontends|3D City Viewer Frontends]]
- [[_COMMUNITY_Project Documentation|Project Documentation]]
- [[_COMMUNITY_A5 Optimization Logs|A5 Optimization Logs]]
- [[_COMMUNITY_Roadmaps and Calibration Integrity|Roadmaps and Calibration Integrity]]
- [[_COMMUNITY_Calibration Pipeline Setup|Calibration Pipeline Setup]]
- [[_COMMUNITY_MAPPO V2 Layout Config|MAPPO V2 Layout Config]]
- [[_COMMUNITY_Signal Phase Padding Utility|Signal Phase Padding Utility]]
- [[_COMMUNITY_Signal Link Padding Utility|Signal Link Padding Utility]]
- [[_COMMUNITY_Corridor Policy Step|Corridor Policy Step]]
- [[_COMMUNITY_V2 Checkpoint Validation|V2 Checkpoint Validation]]
- [[_COMMUNITY_Policy Sampling Utility|Policy Sampling Utility]]
- [[_COMMUNITY_Policy Evaluation Utility|Policy Evaluation Utility]]
- [[_COMMUNITY_Training Plotter|Training Plotter]]
- [[_COMMUNITY_TraCI Signal Controller|TraCI Signal Controller]]
- [[_COMMUNITY_TraCI Signal Labels|TraCI Signal Labels]]
- [[_COMMUNITY_Short-Training Policy Collapse|Short-Training Policy Collapse]]
- [[_COMMUNITY_State-Action Size Fallbacks|State-Action Size Fallbacks]]
- [[_COMMUNITY_ATLAS V1 Architecture Baseline|ATLAS V1 Architecture Baseline]]
- [[_COMMUNITY_Shadow Mode Evaluation|Shadow Mode Evaluation]]
- [[_COMMUNITY_Signal Control Baselines|Signal Control Baselines]]
- [[_COMMUNITY_Bellevue City Engagement|Bellevue City Engagement]]
- [[_COMMUNITY_Regime B Optimization Logs|Regime B Optimization Logs]]

## God Nodes (most connected - your core abstractions)
1. `MultiTlsEnv` - 44 edges
2. `DQNAgent` - 30 edges
3. `SumoTrafficEnv` - 30 edges
4. `CoLightGAT` - 28 edges
5. `FRAPEncoder` - 26 edges
6. `SharedActor` - 26 edges
7. `CentralCritic` - 25 edges
8. `DRWrapper` - 19 edges
9. `MAPPOTrainer` - 18 edges
10. `Path` - 17 edges

## Surprising Connections (you probably didn't know these)
- `PyGame to SUMO Translator` --semantically_similar_to--> `PyGame Traffic Control Visualizer`  [INFERRED] [semantically similar]
  PyGame_to_SUMO.py → SUMO/v2/ai/visualize_sumo_ai.py
- `main()` --calls--> `DQNAgent`  [INFERRED]
  SUMO/v2/ai/sanity_check.py → SUMO/v2/ai/dqn_agent.py
- `int` --uses--> `MultiTlsEnv`  [INFERRED]
  SUMO/v2/ai/v2/domain_randomization.py → SUMO/v2/ai/multi_env.py
- `bool` --uses--> `MultiTlsEnv`  [INFERRED]
  SUMO/v2/ai/v2/domain_randomization.py → SUMO/v2/ai/multi_env.py
- `ndarray` --uses--> `MultiTlsEnv`  [INFERRED]
  SUMO/v2/ai/v2/domain_randomization.py → SUMO/v2/ai/multi_env.py

## Hyperedges (group relationships)
- **WebSocket Simulation Communication Flow** — run_websocket_ai, run_websocket_sim, websocket_server [INFERRED 0.95]
- **Multi-Intersection RL Framework** — multi_env, train_multi_dqn, eval_network, regression_test [INFERRED 0.95]
- **Single-Intersection RL Pipeline** — sumo_env, train_dqn_sumo, eval, sanity_check [INFERRED 0.95]
- **V2 MAPPO Neural Modules** — frap_encoder_FRAPEncoder, colight_gat_CoLightGAT, shared_policy_SharedActor, centralized_critic_CentralCritic [EXTRACTED 1.00]
- **SUMO Calibration Pipeline** — build_calibrated_routes_py, calibrate_carfollow_py, concept_calibration_provenance, concept_krauss_calibration [EXTRACTED 1.00]
- **V2 Inference Pipeline** — inference_adapter_V2CorridorPolicy, live_inference_V2InferenceLoop, shared_policy_SharedActor [EXTRACTED 1.00]
- **DQN Reward Ablations** — readme_max_pressure_reward, readme_combined_reward, readme_differential_reward, readme_anti_starve_reward [INFERRED 0.95]
- **ATLAS V2 Neural Signal Control Stack** — planv2_frap_encoder, planv2_colight_gat, planv2_mappo, planv2_pressure_reward [INFERRED 0.85]
- **3D City Viewer Frontend Variants** — index_3d_city_viewer, index_cyberpunk_viewer, index_perf_optimized_viewer [INFERRED 0.85]
- **Local Perception (Intersection Level) Flow** — flow_basic_sensor, flow_basic_camera, flow_yolo_perception_engine, flow_vehicle_count_density, flow_edge_intelligence, flow_emergency_pedestrian_detection, flow_lpwan_lora_protocol, flow_basic_traffic_cameras_iot_gateway, flow_distributed_database, flow_real_time_optimization [EXTRACTED 1.00]
- **Core Intelligence (The Brain) Flow** — flow_machine_learning_model, flow_historical_trends, flow_predictive_traffic_optimizations, flow_ml_reinforcement_learning_agent, flow_adaptive_traffic_signal, flow_adaptive_traffic_system_middle, flow_adaptive_traffic_system_bottom [EXTRACTED 1.00]
- **Core Project Components List** — flow_core_real_time_adaptive_infrastructure, flow_core_iot_driven_connectivity, flow_core_predictive_data_analysis, flow_core_intelligent_management_output, flow_core_mobility_emergency_response [EXTRACTED 1.00]
- **PyTorch Selection Rationale for Traffic Research** — pytorch_ai_reasons_why_its_a_good_choice_pytorch, pytorch_ai_reasons_why_its_a_good_choice_dynamic_computational_graph, pytorch_ai_reasons_why_its_a_good_choice_debugging_transparency, pytorch_ai_reasons_why_its_a_good_choice_research_ecosystem, pytorch_ai_reasons_why_its_a_good_choice_traffic_management_research [EXTRACTED 1.00]

## Communities (51 total, 21 thin omitted)

### Community 0 - "PyTorch Device and Tensor Utils"
Cohesion: 0.07
Nodes (50): device, int, Tensor, bool, int, Tensor, int, Tensor (+42 more)

### Community 1 - "Network Evaluation and Baseline Comparison"
Cohesion: 0.06
Nodes (48): Corridor Adjacency Graph, fixed_actions_factory(), load_coordinated_agents(), load_coordinated_agents_v2(), main(), parse_args(), Network-level evaluation: coordinated DQN vs fair baselines.  Compares three p, Per-TLS round-robin: each light advances one green slot per decision. (+40 more)

### Community 2 - "SUMO Environment State Utilities"
Cohesion: 0.09
Nodes (27): action_size(), arrived_total(), cumulative_wait(), departed_total(), _is_green(), _next_label(), Single-intersection RL environment for SUMO via TraCI.  Action semantics ----, Open SUMO once headlessly to read TLS structure, then close. (+19 more)

### Community 3 - "Multi-TLS Environment and Domain Randomization"
Cohesion: 0.08
Nodes (26): MultiTlsEnv, bool, float, int, ndarray, float, int, DRWrapper (+18 more)

### Community 4 - "Double-DQN Agent"
Cohesion: 0.09
Nodes (25): DQNAgent, load_for_inference(), Double-DQN agent with target network and uniform replay buffer., Double-DQN over a small MLP defined in ``traffic_base.BaseTrafficAI``., ReplayBuffer, BaseTrafficAI, linear_epsilon(), main() (+17 more)

### Community 5 - "WebSocket AI Server"
Cohesion: 0.07
Nodes (26): int, str, bool, str, int, str, _build_frap_state_for_tls(), V2 inference loop for the live WebSocket runner.  run_websocket_ai.py keeps it (+18 more)

### Community 6 - "SUMO Junction Graph Nodes"
Cohesion: 0.05
Nodes (36): 3153556582, downstream, upstream, 53141735, downstream, upstream, 53254289, downstream (+28 more)

### Community 7 - "FRAP State and Adjacency"
Cohesion: 0.10
Nodes (17): frap_max_movements(), frap_p_max(), load_adjacency(), MultiTlsEnv, Multi-intersection environment: one SUMO process, N coordinated agents.  ``Mul, Advance the shared sim one second; update network accumulators.         Returns, [incoming_queue_norm, pressure_norm, green_progress] for a         neighbour TL, Fan the curriculum reward weights out to every unit. Called by         the trai (+9 more)

### Community 8 - "SUMO Calibration Data Loading"
Cohesion: 0.11
Nodes (32): main(), Plot training curves from ``ai/logs/train_log.csv``.  Run from ``SUMO/v2``:, Path, build(), collect_records(), compass_dir(), incoming_edges_by_compass(), IntersectionRecord (+24 more)

### Community 9 - "ATLAS V2 Architecture Components"
Cohesion: 0.13
Nodes (16): CentralCritic, CoLightGAT, Per-Minibatch Advantage Normalization, CoLight Graph Attention, Centralized Training with Decentralized Execution (CTDE), Domain Randomization, FRAP Phase-Symmetry-Invariant Encoder, Corridor Parameter Sharing (+8 more)

### Community 10 - "Adaptive Traffic Infrastructure"
Cohesion: 0.14
Nodes (22): Adaptive Traffic Signal, Adaptive Traffic System (Bottom), Adaptive Traffic System (Middle), Basic Camera, Basic Sensor, Basic Traffic Cameras (IOT Gateway), Intelligent Management Output, IOT-Driven Connectiivity (+14 more)

### Community 11 - "Academic TSC References"
Cohesion: 0.11
Nodes (22): WSDOT TSMO Preemption Guidance, aUToLights (arXiv 2305.08673), CoLight (arXiv 1905.05717), COMA (arXiv 1705.08926), GEH Distance Distribution Monitor (arXiv 2511.13785), Domain Randomization for TSC (arXiv 2307.11357), Edge-AI Perception (arXiv 2601.07845), FRAP (arXiv 1905.04722) (+14 more)

### Community 12 - "Regression Gate Tests"
Cohesion: 0.22
Nodes (19): gate_against_baseline(), _gate_metric(), _git_sha(), main(), parse_args(), Phase 0 regression net: does V1 still win?  A single command answering "did an, Run the V1 coordinated DQN on each seed. Returns one metrics dict     per seed, Returns (passed: bool, lines: list[str]) for one metric.      Two checks: (+11 more)

### Community 13 - "Krauss Car-Following Calibration"
Cohesion: 0.23
Nodes (14): grid_search_against_target(), KraussParams, main(), parse_args(), Emit a calibrated Krauss vTypes XML; optional grid-search if sat-flow lands., Write the additional-files-compatible vType block. Provenance     lives in XML, Run SUMO for ``sim_duration`` seconds with the given vType     swapped in, meas, Brute-force best (tau, sigma, decel) against the measured target     discharge (+6 more)

### Community 14 - "Traffic Signal Policy Baselines"
Cohesion: 0.19
Nodes (14): actuated_policy(), fixed_policy_factory(), _fmt_mean_std(), main(), parse_args(), Compare the trained DQN agent against baselines.  Four policies are compared o, Roll out one episode with the DQN-env controlling the TLS., Roll out one episode letting SUMO drive the TLS from the .net.xml. (+6 more)

### Community 15 - "Training Ablation Logs"
Cohesion: 0.15
Nodes (13): Ablation Study Results Log, Baseline vs DQN Performance Log, Anti-Starve Training Console Log, Combined Training Console Log, Differential Training Console Log, Max Pressure Training Console Log, MPLight (AAAI 2020), PressLight (KDD 2019) (+5 more)

### Community 16 - "Corridor Adjacency Builder"
Cohesion: 0.25
Nodes (10): build_adjacency(), _build_node_tls_map(), main(), _nearest_tls(), Offline corridor-adjacency builder.  Walks the road graph of a SUMO ``.net.xml, Junction node ids this TLS physically controls.      ``net.getNode(tls_id)`` K, node id -> tls id (a junction belongs to at most one TLS)., BFS over the road graph from ``start_edges`` until the first edge     incident (+2 more)

### Community 17 - "MAPPO Shared Actor Policy"
Cohesion: 0.24
Nodes (7): bool, Tensor, evaluate_actions(), Parameter-shared actor head with masked-categorical action output.  Sits at th, Per-phase masked logits for a single TLS.          Args:             light_em, Batched over the corridor.          Args:             light_embeddings: (N_tl, sample_actions()

### Community 18 - "SUMO GUI Simulation Launcher"
Cohesion: 0.31
Nodes (8): launch_sumo_gui(), main(), SUMO Simulation Launcher ======================== Single entry-point to select, Rewrite sim.sumocfg to use the given route file., Run another Python script in the v2 directory., Open SUMO-GUI with sim.sumocfg, optionally overriding the route file., run_script(), set_route_file()

### Community 19 - "PyTorch Selection Rationale"
Cohesion: 0.52
Nodes (7): PyTorch Autograd, Debugging and Development Transparency, Dynamic Computational Graph Architecture, PyTorch, Research and Experimentation Ecosystem, torch.nn Module, Traffic Management Research

### Community 20 - "Krauss Calibration Scripts"
Cohesion: 0.40
Nodes (4): KraussParams, Calibration Provenance Tracking, Krauss Model Calibration, sumo_calibration/__init__.py

### Community 21 - "3D Viewer WebSockets Frontend"
Cohesion: 0.33
Nodes (6): connectWebSocket, handleStep, sendSetPhase, updateAiBadge, updateTLPanel, wsLog

### Community 22 - "Real-Life Simulation Data Processing"
Cohesion: 0.33
Nodes (4): load_excel_data(), normalize_name(), Aggressively simplifies names to match 'Way Northeast' style maps, Loads XLSX and cleans headers/values to avoid whitespace errors

### Community 23 - "3D City Viewer Frontends"
Cohesion: 0.50
Nodes (5): Double-DQN Controller Integration, 3D City Viewer Frontend Portal, Cyberpunk 3D City Viewer Variant, Performance-Optimized 3D City Viewer Variant, V1 Double-DQN Agent Definition

## Knowledge Gaps
- **63 isolated node(s):** `downstream`, `upstream`, `downstream`, `upstream`, `downstream` (+58 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **21 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `MultiTlsEnv` connect `FRAP State and Adjacency` to `PyTorch Device and Tensor Utils`, `Network Evaluation and Baseline Comparison`, `SUMO Environment State Utilities`, `Multi-TLS Environment and Domain Randomization`, `Double-DQN Agent`, `SUMO Calibration Data Loading`, `Regression Gate Tests`?**
  _High betweenness centrality (0.173) - this node is a cross-community bridge._
- **Why does `Path` connect `SUMO Calibration Data Loading` to `PyTorch Device and Tensor Utils`, `Network Evaluation and Baseline Comparison`, `Double-DQN Agent`, `WebSocket AI Server`, `FRAP State and Adjacency`, `Regression Gate Tests`, `Krauss Car-Following Calibration`, `Traffic Signal Policy Baselines`?**
  _High betweenness centrality (0.130) - this node is a cross-community bridge._
- **Why does `SumoTrafficEnv` connect `SUMO Environment State Utilities` to `Double-DQN Agent`, `Traffic Signal Policy Baselines`, `FRAP State and Adjacency`?**
  _High betweenness centrality (0.097) - this node is a cross-community bridge._
- **Are the 26 inferred relationships involving `MultiTlsEnv` (e.g. with `MultiTlsEnv` and `str`) actually correct?**
  _`MultiTlsEnv` has 26 INFERRED edges - model-reasoned connections that need verification._
- **Are the 23 inferred relationships involving `DQNAgent` (e.g. with `str` and `bool`) actually correct?**
  _`DQNAgent` has 23 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `SumoTrafficEnv` (e.g. with `SumoTrafficEnv` and `Namespace`) actually correct?**
  _`SumoTrafficEnv` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 22 inferred relationships involving `CoLightGAT` (e.g. with `_Modules` and `V2CorridorPolicy`) actually correct?**
  _`CoLightGAT` has 22 INFERRED edges - model-reasoned connections that need verification._