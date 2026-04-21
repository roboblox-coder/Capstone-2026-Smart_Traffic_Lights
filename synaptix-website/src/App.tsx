/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState } from 'react';
import { 
  Brain, 
  Activity, 
  BarChart3, 
  Cpu,
  Zap,
  Target
} from 'lucide-react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  AreaChart,
  Area,
  BarChart,
  Bar,
  Legend
} from 'recharts';
import cyberpunkBg from './assets/cyberpunk.jpg';

const TrafficLight = (props: React.SVGProps<SVGSVGElement>) => (
  <svg
    {...props}
    xmlns="http://www.w3.org/2000/svg"
    width="24"
    height="24"
    viewBox="0 0 24 24"
    fill="none"
    stroke="currentColor"
    strokeWidth="2"
    strokeLinecap="round"
    strokeLinejoin="round"
  >
    <rect x="8" y="2" width="8" height="20" rx="2" />
    <circle cx="12" cy="7" r="2" />
    <circle cx="12" cy="12" r="2" />
    <circle cx="12" cy="17" r="2" />
  </svg>
);

const Github = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"></path></svg>
);

const Linkedin = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M16 8a6 6 0 0 1 6 6v7h-4v-7a2 2 0 0 0-2-2 2 2 0 0 0-2 2v7h-4v-7a6 6 0 0 1 6-6z"></path><rect x="2" y="9" width="4" height="12"></rect><circle cx="4" cy="4" r="2"></circle></svg>
);

const Download = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>
);

const Menu = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="3" y1="12" x2="21" y2="12"></line><line x1="3" y1="6" x2="21" y2="6"></line><line x1="3" y1="18" x2="21" y2="18"></line></svg>
);

const X = (props: React.SVGProps<SVGSVGElement>) => (
  <svg {...props} xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"><line x1="18" y1="6" x2="6" y2="18"></line><line x1="6" y1="6" x2="18" y2="18"></line></svg>
);

function App() {
  const [activeTab, setActiveTab] = useState('Simulation Introduction');
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <div className="relative w-full min-h-screen bg-[#020202] text-white overflow-x-hidden font-sans">
      <style>{`
        @keyframes fadeSlideInRight {
          from { opacity: 0; transform: translateX(-30px); }
          to { opacity: 1; transform: translateX(0); }
        }
        @keyframes fadeSlideInDown {
          from { opacity: 0; transform: translateY(-20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        @keyframes fadeSlideInUp {
          from { opacity: 0; transform: translateY(20px); }
          to { opacity: 1; transform: translateY(0); }
        }
        .animate-tab-content {
          animation: fadeSlideInRight 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .animate-header-logo {
          animation: fadeSlideInRight 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
        }
        .animate-header-nav {
          animation: fadeSlideInDown 0.8s cubic-bezier(0.16, 1, 0.3, 1) forwards;
          animation-delay: 0.2s;
          opacity: 0;
        }
        .animate-team-card {
          animation: fadeSlideInUp 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
          opacity: 0;
        }
      `}</style>

      <div className="fixed inset-0 z-0">
        <img
          src={cyberpunkBg}
          alt="Cyberpunk city background"
          className="w-full h-full object-cover opacity-60"
        />
        <div className="absolute inset-0 bg-gradient-to-b from-[#020202]/50 via-transparent to-[#020202]/90"></div>
      </div>

      <div className="relative z-10 w-full min-h-screen pointer-events-none flex flex-col justify-between">
        <header className="p-6 md:p-10 flex justify-between items-center">
          <div
            className="flex items-center gap-4 group cursor-pointer pointer-events-auto animate-header-logo"
            onClick={() => {
              setActiveTab('Simulation Introduction');
              setIsMenuOpen(false);
            }}
          >
            <div className="relative">
              <div className="absolute inset-0 bg-[#FFD700] blur-lg opacity-20 group-hover:opacity-50 transition-opacity"></div>
              <div className="relative w-10 h-10 bg-[#FFD700] rounded-lg flex items-center justify-center shadow-[0_0_20px_rgba(255,215,0,0.4)] group-hover:shadow-[0_0_30px_rgba(255,215,0,0.6)] transition-all transform group-hover:scale-110">
                <TrafficLight className="w-6 h-6 text-black" />
              </div>
            </div>
            <div className="flex flex-col">
              <h1 className="text-2xl md:text-3xl font-black italic tracking-[0.15em] text-border leading-none">SYNAPTIX</h1>
              <span className="text-[8px] md:text-[10px] font-mono tracking-[0.4em] text-[#FFD700] mt-1 opacity-80">ARTIFICIAL INTELLIGENCE</span>
            </div>
          </div>

          <nav className="hidden lg:flex items-center gap-2 p-1 bg-white/5 backdrop-blur-md rounded-2xl border border-white/10 animate-header-nav">
            {['Simulation Introduction', 'Dashboard', 'Project Architecture', 'Meet the Team'].map((tab) => (
              <button
                key={tab}
                onClick={() => setActiveTab(tab)}
                className={`px-6 py-2 rounded-xl text-xs font-bold tracking-widest uppercase transition-all duration-300 pointer-events-auto relative overflow-hidden group ${
                  activeTab === tab
                    ? 'text-black bg-[#FFD700] shadow-[0_0_15px_rgba(255,215,0,0.3)]'
                    : 'text-gray-400 hover:text-white hover:bg-white/5'
                }`}
              >
                <span className="relative z-10">{tab}</span>
                {activeTab !== tab && (
                  <div className="absolute bottom-0 left-0 w-full h-0.5 bg-[#FFD700] transform scale-x-0 group-hover:scale-x-100 transition-transform origin-left duration-300" />
                )}
              </button>
            ))}
          </nav>

          <button
            className="lg:hidden p-3 bg-white/5 backdrop-blur-md rounded-xl border border-white/10 text-[#FFD700] pointer-events-auto transition-all hover:bg-white/10"
            onClick={() => setIsMenuOpen(!isMenuOpen)}
          >
            {isMenuOpen ? <X className="w-6 h-6" /> : <Menu className="w-6 h-6" />}
          </button>

          <div className="w-[200px] hidden lg:block"></div>
        </header>

        <div className={`fixed inset-0 z-50 lg:hidden transition-all duration-500 ${isMenuOpen ? 'opacity-100 pointer-events-auto' : 'opacity-0 pointer-events-none'}`}>
          <div className="absolute inset-0 bg-[#020202]/95 backdrop-blur-xl"></div>
          <div className="relative h-full flex flex-col p-8">
            <div className="flex justify-between items-center mb-16">
              <div className="flex items-center gap-4">
                <div className="w-10 h-10 bg-[#FFD700] rounded-lg flex items-center justify-center">
                  <TrafficLight className="w-6 h-6 text-black" />
                </div>
                <h1 className="text-2xl font-black italic tracking-[0.15em] text-border">SYNAPTIX</h1>
              </div>
              <button
                className="p-3 bg-white/5 rounded-xl border border-white/10 text-[#FFD700]"
                onClick={() => setIsMenuOpen(false)}
              >
                <X className="w-6 h-6" />
              </button>
            </div>

            <nav className="flex flex-col gap-6">
              {['Simulation Introduction', 'Dashboard', 'Project Architecture', 'Meet the Team'].map((tab, i) => (
                <button
                  key={tab}
                  onClick={() => {
                    setActiveTab(tab);
                    setIsMenuOpen(false);
                  }}
                  className={`text-left py-4 border-b border-white/5 transition-all duration-300 ${
                    activeTab === tab ? 'text-[#FFD700] text-3xl font-bold italic' : 'text-gray-500 text-2xl font-medium'
                  }`}
                  style={{ transitionDelay: `${i * 50}ms` }}
                >
                  {tab}
                </button>
              ))}
            </nav>
          </div>
        </div>

        <main className="flex-1 flex flex-col justify-center items-center text-center py-16 px-6 md:px-20 max-w-7xl mx-auto w-full">
          {activeTab === 'Simulation Introduction' && (
            <div key="intro" className="max-w-2xl animate-tab-content flex flex-col items-center">
              <h2 className="text-5xl md:text-7xl font-bold leading-tight mb-6 tracking-tight text-border">
                Meet the <br/>
                <span className="text-white">
                  Next-Generation in Urban Mobility
                </span> <br/>
              </h2>
             
              <p className="text-lg text-white-400 mb-10 max-w-xl leading-relaxed text-border mx-auto">
                Synaptix AI bridges the gap between complex SUMO traffic simulations and advanced neural network architectures. Linked below are both the instructions to install the main files (bottom left) and the AI aspects (bottom right)
              </p>

              <div className="flex flex-wrap gap-4 pointer-events-auto justify-center">
                <button
                  onClick={() => window.open('https://github.com/roboblox-coder/Capstone-2026-Smart_Traffic_Lights/tree/main/SUMO', '_blank')}
                  className="flex items-center gap-2 px-8 py-4 bg-[#FFD700] text-black rounded-xl font-semibold hover:bg-[#FFC800] transition-colors shadow-[0_0_20px_rgba(255,215,0,0.3)] hover:shadow-[0_0_30px_rgba(255,215,0,0.5)]"
                >
                  <Download className="w-4 h-4" />
                  DOWNLOAD & INSTALL GUIDE
                </button>
                <button
                  onClick={() => window.open('https://github.com/roboblox-coder/Capstone-2026-Smart_Traffic_Lights/blob/Deltazorka-patch-1/README.md', '_blank')}
                  className="flex items-center gap-2 px-8 py-4 bg-[#FFD700] text-black rounded-xl font-semibold hover:bg-[#FFC800] transition-colors shadow-[0_0_20px_rgba(255,215,0,0.3)] hover:shadow-[0_0_30px_rgba(255,215,0,0.5)] pointer-events-auto"
                >
                  <Download className="w-4 h-4" />
                  AI INSTALL GUIDE
                </button>
              </div>
            </div>
          )}

          {activeTab === 'Dashboard' && (
            <div key="dashboard" className="max-w-6xl w-full animate-tab-content flex flex-col items-center">
              <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 w-full mb-8 pointer-events-auto">
                {/* AI Model Info Card */}
                <div className="lg:col-span-2 p-8 rounded-3xl bg-black/60 border border-white/10 backdrop-blur-md text-left">
                  <div className="flex items-center gap-3 mb-6">
                    <div className="p-2 bg-[#FFD700]/10 rounded-lg border border-[#FFD700]/20">
                      <Brain className="w-6 h-6 text-[#FFD700]" />
                    </div>
                    <h3 className="text-2xl font-bold text-white">AI Model Architecture</h3>
                  </div>
                  
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="space-y-4">
                      <div className="flex items-start gap-3">
                        <Cpu className="w-5 h-5 text-[#FFD700] mt-1 shrink-0" />
                        <div>
                          <p className="text-sm font-bold text-white uppercase tracking-wider">Neural Network</p>
                          <p className="text-gray-400 text-sm">Deep Q-Network (DQN) with Experience Replay and Target Network stabilization.</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <Zap className="w-5 h-5 text-[#FFD700] mt-1 shrink-0" />
                        <div>
                          <p className="text-sm font-bold text-white uppercase tracking-wider">Input State</p>
                          <p className="text-gray-400 text-sm">Traffic density, vehicle speeds, and queue lengths per lane from SUMO TraCI.</p>
                        </div>
                      </div>
                    </div>
                    <div className="space-y-4">
                      <div className="flex items-start gap-3">
                        <Target className="w-5 h-5 text-[#FFD700] mt-1 shrink-0" />
                        <div>
                          <p className="text-sm font-bold text-white uppercase tracking-wider">Reward Function</p>
                          <p className="text-gray-400 text-sm">Weighted optimization of throughput vs. average waiting time reduction.</p>
                        </div>
                      </div>
                      <div className="flex items-start gap-3">
                        <Activity className="w-5 h-5 text-[#FFD700] mt-1 shrink-0" />
                        <div>
                          <p className="text-sm font-bold text-white uppercase tracking-wider">Action Space</p>
                          <p className="text-gray-400 text-sm">Dynamic phase switching and adaptive green-time allocation.</p>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Stats Summary Card */}
                <div className="p-8 rounded-3xl bg-black/60 border border-white/10 backdrop-blur-md text-left flex flex-col justify-between">
                  <div>
                    <div className="flex items-center gap-3 mb-6">
                      <div className="p-2 bg-[#FFD700]/10 rounded-lg border border-[#FFD700]/20">
                        <BarChart3 className="w-6 h-6 text-[#FFD700]" />
                      </div>
                      <h3 className="text-2xl font-bold text-white">Training Stats</h3>
                    </div>
                    <div className="space-y-6">
                      <div>
                        <p className="text-xs font-mono text-gray-500 uppercase tracking-widest mb-1">Total Episodes</p>
                        <p className="text-4xl font-black text-[#FFD700]">1,250</p>
                      </div>
                      <div>
                        <p className="text-xs font-mono text-gray-500 uppercase tracking-widest mb-1">Avg. Reward</p>
                        <p className="text-4xl font-black text-white">+84.2</p>
                      </div>
                      <div>
                        <p className="text-xs font-mono text-gray-500 uppercase tracking-widest mb-1">Success Rate</p>
                        <p className="text-4xl font-black text-white">92.4%</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              {/* Charts Section */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full pointer-events-auto">
                <div className="p-8 rounded-3xl bg-black/60 border border-white/10 backdrop-blur-md text-left h-[400px]">
                  <h4 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                    <Activity className="w-4 h-4 text-[#FFD700]" />
                    Reward Convergence
                  </h4>
                  <ResponsiveContainer width="100%" height="85%">
                    <AreaChart
                      data={[
                        { episode: 0, reward: -150 },
                        { episode: 100, reward: -80 },
                        { episode: 200, reward: -20 },
                        { episode: 300, reward: 10 },
                        { episode: 400, reward: 45 },
                        { episode: 500, reward: 60 },
                        { episode: 600, reward: 72 },
                        { episode: 700, reward: 78 },
                        { episode: 800, reward: 82 },
                        { episode: 900, reward: 84 },
                        { episode: 1000, reward: 85 },
                      ]}
                    >
                      <defs>
                        <linearGradient id="colorReward" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="5%" stopColor="#FFD700" stopOpacity={0.3}/>
                          <stop offset="95%" stopColor="#FFD700" stopOpacity={0}/>
                        </linearGradient>
                      </defs>
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                      <XAxis dataKey="episode" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '8px' }}
                        itemStyle={{ color: '#FFD700' }}
                      />
                      <Area type="monotone" dataKey="reward" stroke="#FFD700" fillOpacity={1} fill="url(#colorReward)" strokeWidth={3} />
                    </AreaChart>
                  </ResponsiveContainer>
                </div>

                <div className="p-8 rounded-3xl bg-black/60 border border-white/10 backdrop-blur-md text-left h-[400px]">
                  <h4 className="text-lg font-bold text-white mb-6 flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-[#FFD700]" />
                    Traffic Throughput (Veh/Hr)
                  </h4>
                  <ResponsiveContainer width="100%" height="85%">
                    <BarChart
                      data={[
                        { time: '08:00', baseline: 1200, synaptix: 1450 },
                        { time: '10:00', baseline: 900, synaptix: 1100 },
                        { time: '12:00', baseline: 1100, synaptix: 1300 },
                        { time: '14:00', baseline: 1000, synaptix: 1250 },
                        { time: '16:00', baseline: 1500, synaptix: 1850 },
                        { time: '18:00', baseline: 1400, synaptix: 1700 },
                      ]}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                      <XAxis dataKey="time" stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <YAxis stroke="#666" fontSize={12} tickLine={false} axisLine={false} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#111', border: '1px solid #333', borderRadius: '8px' }}
                      />
                      <Legend verticalAlign="top" align="right" iconType="circle" />
                      <Bar dataKey="baseline" fill="#444" radius={[4, 4, 0, 0]} name="Standard Logic" />
                      <Bar dataKey="synaptix" fill="#FFD700" radius={[4, 4, 0, 0]} name="Synaptix AI" />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'Project Architecture' && (
            <div key="architecture" className="max-w-4xl w-full animate-tab-content flex flex-col items-center">
              <h2 className="text-4xl md:text-6xl font-bold leading-tight mb-8 tracking-tight text-border">
                SUMO Integration <br/>
                <span className="text-[#FFD700]">Architecture</span>
              </h2>
             
              <div className="bg-black/60 border border-white/10 rounded-2xl p-4 md:p-8 backdrop-blur-xl font-mono text-[10px] sm:text-xs md:text-base overflow-x-auto custom-scrollbar w-full text-left pointer-events-auto">
                <pre className="text-gray-300 leading-relaxed whitespace-pre min-w-max">
{`SUMO/
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .gitignore
└── v2/
    ├── run.py                # 🚀 Main launcher (start here)
    ├── real_life_simulation.py  # Converts Harvard Excel → SUMO routes
    ├── run_auto_sim.py       # Headless sim + CSV export
    ├── run_traci_sim.py      # TraCI-based sim with AI hooks
    ├── run_websocket_sim.py  # TraCI + WebSocket broadcaster
    ├── websocket_server.py   # Reusable WS server module
    ├── sim.sumocfg           # SUMO config file
    ├── NE_8th_St_Corridor.net.xml  # Road network
    ├── e2output.xml
    ├── harvard_simulation.rou.xml  # Routes from Harvard data
    ├── synthetic.rou.xml     # Synthetic routes
    ├── Real_intersection_data/     # Harvard Excel source files
    ├── ai/                   # This AI helps you set up and run the 3D traffic simulation website using the files from the main branch.
        ├── sumo_env
        ├── traffic_base
        ├── train_dqn_sumo
        └── visualize_sumo_ai
    └── frontend/
        └── index.html (simulation.html on the website)  # Three.js live viewer (no build step)`}
                </pre>
              </div>
            </div>
          )}

          {activeTab === 'Meet the Team' && (
            <div key="team" className="max-w-5xl w-full animate-tab-content flex flex-col items-center">
              <h2 className="text-3xl md:text-5xl font-black italic leading-tight mb-12 tracking-tight text-border">
                Brought to you by:
              </h2>
             
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 lg:gap-8 pointer-events-auto">
                {[
                  {
                    name: "Duke Schnepf",
                    role: "Team Lead & SUMO Architect",
                    description: "Driving the core simulation logic and traci integration for real-time traffic optimization.",
                    linkedin: "https://www.linkedin.com/in/dukeschnepf/",
                    github: "https://github.com/DukeSchnepf"
                  },
                  {
                    name: "Nikolaus Henkel",
                    role: "Visual Simulation Architect",
                    description: "Crafting immersive environments and visual data representations for urban mobility.",
                    linkedin: "https://www.linkedin.com/in/nikolaus-henkel-a67023388/",
                    github: "https://github.com/roboblox-coder"
                  },
                  {
                    name: "Cory Maccini",
                    role: "Data Analyst & UI Designer",
                    description: "Creating website interfaces for optimum engagement.",
                    linkedin: "https://www.linkedin.com/in/cory-m-636320345/",
                    github: "https://github.com/LordPolloRex"
                  },
                  {
                    name: "Ryan Liang",
                    role: "AI Model Trainer",
                    description: "Developing the PyTorch AI to predict and manage urban traffic flow.",
                    linkedin: "https://www.linkedin.com/in/ryan-liang-0753b8387/",
                    github: "https://github.com/Deltazorka"
                  }
                ].map((person, idx) => (
                  <div
                    key={person.name}
                    className="group relative p-8 rounded-3xl bg-black/60 border border-white/10 backdrop-blur-md transition-all overflow-hidden animate-team-card text-left"
                    style={{ animationDelay: `${0.2 + (idx * 0.1)}s` }}
                  >
                   
                    <div className="flex flex-col gap-6 relative z-10">
                      <div>
                        <h3 className="text-2xl font-bold text-white group-hover:text-[#FFD700] transition-colors duration-300 tracking-tight">{person.name}</h3>
                        <p className="text-[#FFD700] font-mono text-xs uppercase tracking-[0.2em] mt-1 opacity-80">{person.role}</p>
                        <p className="text-gray-400 text-sm mt-4 leading-relaxed line-clamp-2 group-hover:line-clamp-none transition-all duration-300">
                          {person.description}
                        </p>
                      </div>

                      <div className="pt-4 border-t border-white/5 flex items-center justify-between">
                        <span className="text-[10px] font-mono text-gray-500 uppercase tracking-widest">Connect</span>
                        <div className="flex gap-8">
                          <button
                            onClick={() => window.open(person.linkedin, '_blank')}
                            className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center hover:bg-[#FFD700] hover:text-black transition-colors duration-300"
                          >
                            <Linkedin className="w-4 h-4" />
                          </button>
                          <button
                            onClick={() => window.open(person.github, '_blank')}
                            className="w-8 h-8 rounded-lg bg-white/5 flex items-center justify-center hover:bg-[#FFD700] hover:text-black transition-colors duration-300"
                          >
                            <Github className="w-4 h-4" />
                          </button>
                        </div>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}

export default App; 