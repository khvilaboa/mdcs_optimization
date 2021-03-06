{3.2.0.3011}
{Enterprise Dynamics startup information}

if(StartingED, SoftStartED([]));


{Model information}

AddLayer([Stands], 3, 15, 0);
AddLayer([Main], 3, 1, 1);
AddLayer([DoNotDraw], 10, 2, 2);
AddLayer([Transportation], 3, 17, 3);
AddLayer([Infrastructure], 3, 9, 4);
AddLayer([Output], 3, 14, 5);
AddLayer([ECMNetwork], 27, 4, 6);
AddLayer([Openings], 3, 8, 7);
AddLayer([Agents], 3, 11, 8);
AddLayer([WalkableAreas], 3, 7, 9);
AddLayer([Activities], 3, 10, 10);
AddLayer([LocalObstacles], 3, 13, 11);
AddLayer([Obstacles], 3, 6, 12);
AddLayer([HeightLayers], 3, 5, 13);
AddLayer([UserActions], 3, 12, 14);
AddLayer([PD_Environment], 27, 3, 15);
AddLayer([StandsActivity], 3, 16, 16);


{Load required atoms}

int011;
int035([ActivityLocation], pDir([Atoms\ACTIVITIES\ActivityLocation.atm]));
int035([AgentDrawInformation], pDir([Atoms\AGENTS\AgentDrawInformation.atm]));
int035([AgentProfile], pDir([Atoms\AGENTS\AgentProfile.atm]));
int035([HeightLayer], pDir([Atoms\ENVIRONMENT\HeightLayer.atm]));
int035([PD_Environment], pDir([Atoms\ENVIRONMENT\PD_Environment.atm]));
int035([Experiment_Wizard], pDir([Atoms\EXPERIMENT\Experiment_Wizard.atm]));
int035([AgentActivity], pDir([Atoms\INPUT\AgentActivity.atm]));
int035([AgentActivityRoute], pDir([Atoms\INPUT\AgentActivityRoute.atm]));
int035([AgentGenerator], pDir([Atoms\INPUT\AgentGenerator.atm]));
int035([AgentInput], pDir([Atoms\INPUT\AgentInput.atm]));
int035([PD_Input], pDir([Atoms\INPUT\PD_Input.atm]));
int035([SimControl], pDir([Atoms\INPUT\SimControl.atm]));
int035([SlopeSpeed], pDir([Atoms\INPUT\SlopeSpeed.atm]));
int035([ActivityRoute], pDir([Atoms\OUTPUT\ActivityRoute.atm]));
int035([AgentStatistics], pDir([Atoms\OUTPUT\AgentStatistics.atm]));
int035([DensityNorms], pDir([Atoms\OUTPUT\DensityNorms.atm]));
int035([FrequencyNorms], pDir([Atoms\OUTPUT\FrequencyNorms.atm]));
int035([lstOutputActivityRoutes], pDir([Atoms\OUTPUT\lstOutputActivityRoutes.atm]));
int035([OutputLayer], pDir([Atoms\OUTPUT\OutputLayer.atm]));
int035([PD_Output], pDir([Atoms\OUTPUT\PD_Output.atm]));
int035([ResultPlayer], pDir([Atoms\OUTPUT\ResultPlayer.atm]));
int035([TravelTimeNorms], pDir([Atoms\OUTPUT\TravelTimeNorms.atm]));
int035([CameraPositions], pDir([Atoms\TOOLS\CameraPositions.atm]));
int035([DisplayAtom], pDir([Atoms\TOOLS\DisplayAtom.atm]));
int035([MovieRecord], pDir([Atoms\TOOLS\MovieRecord.atm]));
int035([RunClock], pDir([Atoms\TOOLS\RunClock.atm]));
int035([SelectionPolygon], pDir([Atoms\TOOLS\SelectionPolygon.atm]));
int035([UserEvents], pDir([Atoms\TOOLS\UserEvents.atm]));
int035([List], pDir([Atoms\TOOLS\UTILITIES\List.atm]));
int012;


{Atom: PD_Input}

sets;
AtomByName([PD_Input], Main);
if(not(AtomExists), Error([Cannot find mother atom 'PD_Input'. Inheriting from BaseClass.]));
CreateAtom(a, s, [PD_Input], 1, false);
SetAtt([lstTempAgents], 0);
SetExprAtt([InitCode], [0]);
SetAtt([lstTables], 0);
int023([], 0, 530432);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(1);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstTables}

sets;
AtomByName([lstTables], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstTables'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstTables], 1, false);
SetAtt([NrCreated], 0);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(2);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: SimControl}

sets;
AtomByName([SimControl], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SimControl'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [SimControl], 1, false);
SetAtt([CycleTime], 0.2);
SetAtt([UseDensityPlanning], 1);
SetAtt([AvgAgentArea], 0.18);
SetAtt([RebuildECMNetwork], 1);
SetAtt([UNUSED_DensityDistance], 2);
SetAtt([ECMLoaded], 0);
SetExprAtt([PixelMeter], [METERSPERPIXEL]);
SetExprAtt([AgentColorType2D], [AGENT_COLORTYPE_2D]);
SetExprAtt([AgentColorType3D], [AGENT_COLORTYPE_3D]);
SetExprAtt([AgentDrawType2D], [AGENT_DRAWTYPE_2D]);
SetExprAtt([AgentDrawType3D], [AGENT_DRAWTYPE_3D]);
SetAtt([CycleEventScheduled], 0);
SetExprAtt([StopTimeSeconds], [0]);
SetAtt([RunUntilStopTime], 0);
SetAtt([Initialized], 0);
SetAtt([EvacuationMode], 0);
SetAtt([RouteFollowingMethod], 15001);
SetAtt([AvoidCollisions_DensityThreshold], -1);
SetAtt([VisualizeSpeed], 0);
SetAtt([UseMesoDensitySpeed], 1);
int023([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(3);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(3, 5);
int015(0, 64, [`ID`
1
2
3
]);
int015(1, 64, [`Name`
`Normal`
`StairsUp`
`StairsDown`
]);
int015(2, 98, [`FormulaType`
2
2
2
]);
int015(3, 102, [`MaxSpeed`
1.34
0.61
0.694
]);
int015(4, 64, [`dFactor`
1.913
3.722
3.802
]);
int015(5, 64, [`dJam`
5.4
5.4
5.4
]);
SetStatus(0);
int018;


{Atom: AgentInput}

sets;
AtomByName([AgentInput], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentInput'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [AgentInput], 1, false);
SetAtt([lstGenerators], 0);
SetAtt([lstProfiles], 0);
SetAtt([lstActivityRoutes], 0);
SetAtt([lstActivities], 0);
SetAtt([MultiplierMaxValue], 5);
SetAtt([MultiplierStepSize], 0.1);
SetAtt([lstProxies], 0);
SetExprAtt([LastProfileFolder], [4DS[pDir([Settings\])]4DS]);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(4);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstGenerators}

sets;
AtomByName([lstGenerators], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstGenerators'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstGenerators], 1, false);
SetAtt([NrCreated], 1);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(5);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: Default_Generator}

sets;
AtomByName([AgentGenerator], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentGenerator'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Default_Generator], 1, false);
SetAtt([ID], 1);
SetAtt([RepetitiveMode], 2002);
SetAtt([RepeatTime], 0);
SetAtt([OffSetTime], 0);
SetExprAtt([NrTimesToRepeat], [{**Unlimited**} 0]);
SetAtt([DelayTime], 0);
SetAtt([CreationTrigger], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([MaxNumberOfAgents], [{**Unlimited**} -1]);
SetAtt([CurrentMultiplier], 1);
SetAtt([TransportGeneratorID], 0);
SetAtt([TransportGenerator], 0);
SetAtt([ReadDB], 0);
SetAtt([DBConnectString], 0);
SetAtt([TableName], 0);
SetExprAtt([ResetCode], [0]);
SetAtt([tempMaxNumberOfAgents], -1);
SetAtt([tempCurrentMultiplier], 1);
SetAtt([AgentsAsPercOfMax], 0);
int023([], 7168771, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetChannels(1, 1);
SetChannelRanges(1, 1, 1, 255);
int001(6);
SetSize(5, 2, 2);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 7);
int015(0, 64, [`Row`
1
]);
int015(1, 64, [`Creation time`
60
]);
int015(2, 64, [`Nr Agents`
1
]);
int015(3, 64, [`Activity route`
1
]);
int015(4, 64, [`Agent profile`
1
]);
int015(5, 86, [`Creation trigger`
0
]);
int015(6, 72, [`TransportID`
0
]);
int015(7, 95, [`TransportRow`
0
]);
SetStatus(0);
int018;
Up;


{Atom: lstProfiles}

sets;
AtomByName([lstProfiles], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstProfiles'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstProfiles], 1, false);
SetAtt([NrCreated], 1);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(7);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: Default_Profile}

sets;
AtomByName([AgentProfile], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentProfile'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Default_Profile], 1, false);
SetAtt([ID], 1);
SetExprAtt([MaxSpeed], [Triangular(1.35, 0.8, 1.75)]);
SetExprAtt([MinSpeed], [0.06]);
SetExprAtt([Radius], [0.239]);
SetExprAtt([DensityCostWeight], [Uniform(0.5, 1.5)]);
SetExprAtt([DensityPlanDistance], [60]);
SetExprAtt([ReplanFrequency], [40]);
SetExprAtt([HistoryPenalty], [0]);
SetExprAtt([HeuristicMultiplier], [1]);
SetExprAtt([SidePreference], [Uniform(-1, 1)]);
SetExprAtt([SidePreferenceNoise], [0.1]);
SetExprAtt([PreferredClearance], [0.3]);
SetExprAtt([Color], [ColorBlue]);
SetExprAtt([ModelID3D], [0]);
SetExprAtt([DiscomfortWeight], [Uniform(0.5, 1.5)]);
SetExprAtt([FixedSpeedFactor], [0]);
SetAtt([UseShortestPath], 0);
SetExprAtt([PersonalDistance], [0.5]);
SetExprAtt([FieldOfViewAngle], [75]);
SetExprAtt([FieldOfViewCollisionRange], [8]);
SetExprAtt([FieldOfViewDensityRange], [2]);
SetAtt([UseDensityRouting], 1);
SetAtt([UseLimitedDensityPlanDistance], 1);
SetAtt([UseRerouting], 0);
SetExprAtt([RandomExtraEdgeCost], [5]);
SetExprAtt([Tokens], [0]);
SetExprAtt([AvoidanceSidePreference], [{**Right preference**}0.1]);
SetExprAtt([SideClearanceFactor], [0]);
SetExprAtt([MaxShortCutDistance], [0]);
SetExprAtt([SidePreferenceUpdateFactor], [1]);
SetAtt([RoutingMethod], 0);
SetExprAtt([AggressivenessThreshold], [0.25]);
SetTextAtt([Description], []);
int042(0, 0, 1);
int023([], 13158600, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(8);
SetSize(0.4, 0.4, 2);
SetTranslation(-0.2, -0.2, 0);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;


{Atom: lstProxies}

sets;
AtomByName([lstProxies], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstProxies'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstProxies], 1, false);
SetAtt([NrCreated], 0);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(9);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstActivityRoutes}

sets;
AtomByName([lstActivityRoutes], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstActivityRoutes'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstActivityRoutes], 1, false);
SetAtt([NrCreated], 1);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(10);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: Default_Route}

sets;
AtomByName([AgentActivityRoute], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivityRoute'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Default_Route], 1, false);
SetAtt([ID], 1);
int023([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(11);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(2, 2);
int015(0, 0, [``
1
2
]);
int015(1, 0, [1
1
2
]);
int015(2, 0, [2
0
0
]);
SetStatus(0);
int018;
Up;


{Atom: lstAgentActivities}

sets;
AtomByName([lstAgentActivities], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstAgentActivities'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstAgentActivities], 1, false);
SetAtt([NrCreated], 2);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(12);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: Entry}

sets;
AtomByName([AgentActivity], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivity'. Inheriting from BaseClass.]));
CreateAtom(a, s, [Entry], 1, false);
SetAtt([ID], 1);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetTextAtt([ActivityGroup], [***ALL***]);
SetExprAtt([PreCondition], [1]);
SetExprAtt([PostCondition], [1]);
SetExprAtt([LocationDistribution], [LOCATIONDISTRIBUTION_UNIFORM]);
SetAtt([RevisitAllowed], 0);
SetExprAtt([LocationAssignment], [0]);
SetAtt([NumberAssigned], 0);
SetAtt([LastRowNumber], 0);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetExprAtt([ResetCode], [0]);
int023([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(13);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(115, 3);
int015(0, 0, [``
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
]);
int015(1, 0, [`ActivityID`
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
]);
int015(2, 0, [`vtp`
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
]);
int015(3, 0, [`%`
]);
SetStatus(0);
int018;


{Atom: Exit}

sets;
AtomByName([AgentActivity], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentActivity'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Exit], 1, false);
SetAtt([ID], 2);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetTextAtt([ActivityGroup], [***ALL***]);
SetExprAtt([PreCondition], [1]);
SetExprAtt([PostCondition], [1]);
SetExprAtt([LocationDistribution], [LOCATIONDISTRIBUTION_UNIFORM]);
SetAtt([RevisitAllowed], 0);
SetExprAtt([LocationAssignment], [0]);
SetAtt([NumberAssigned], 0);
SetAtt([LastRowNumber], 0);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetExprAtt([ResetCode], [0]);
int023([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(14);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(115, 3);
int015(0, 0, [``
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
]);
int015(1, 0, [`ActivityID`
1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96
97
98
99
100
101
102
103
104
105
106
107
108
109
110
111
112
113
114
115
116
]);
int015(2, 0, [`vtp`
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
0
]);
int015(3, 0, [`%`
]);
SetStatus(0);
int018;
Up;
Up;


{Atom: SlopeSpeed}

sets;
AtomByName([SlopeSpeed], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SlopeSpeed'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [SlopeSpeed], 1, false);
int023([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(15);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(6, 3);
int015(0, 64, [`ID`
1
2
3
4
5
6
]);
int015(1, 64, [`Slope (%)`
0
5
10
15
20
40
]);
int015(2, 98, [`% Speed up`
100
96.3
88.8
79.9
70.9
41
]);
int015(3, 102, [`% Speed down`
100
103
104.5
104.5
103
74.6
]);
SetStatus(0);
int018;


{Atom: Experiment_Wizard}

sets;
AtomByName([Experiment_Wizard], Main);
if(not(AtomExists), Error([Cannot find mother atom 'Experiment_Wizard'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [Experiment_Wizard], 1, false);
SetAtt([NumberScenarios], 1);
SetAtt([CurScen], 0);
SetAtt([NumberOfReps], 1);
SetExprAtt([Runlength], [mins(15)]);
SetExprAtt([CodeAtStart], [0]);
SetExprAtt([CodeAtEnd], [0]);
SetExprAtt([Path], [4DS[ModDir([PD_Results\])]4DS]);
SetAtt([numparams], 0);
SetAtt([executescenario], 1);
SetAtt([LogOutput], 1);
SetAtt([LoadOutputOfLastReplication], 1);
SetAtt([FootstepFile], 0);
SetAtt([LogFootSteps], 1);
SetAtt([LogFootStepFrequency], 1);
SetAtt([Initialized], 0);
SetExprAtt([AddTextToFolderName], [4DS[[]]4DS]);
int023([], 0, 525480);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetChannels(1, 0);
SetChannelRanges(0, 1, 0, 0);
int001(16);
int013(1, 0, true, true, 5, 0, [Optional connection to Central Channel of 70 MB Experiment Atom], []);
SetSize(11, 2, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 9);
int015(0, 64, [`Seed_Value`
1
]);
int015(1, 95, [`Scen nr`
`Experiment1`
]);
int015(2, 93, [4DS[`pad`
`ModDir([PD_Results\])`
]4DS]);
int015(3, 75, [`Omschrijving`
`--- describe new scenario ---`
]);
int015(4, 90, [`NumReps`
1
]);
int015(5, 79, [`Runlengte`
`mins(15)`
]);
int015(6, 64, [`StartCode`
0
]);
int015(7, 64, [`EndCode`
0
]);
int015(8, 64, [`Execute`
1
]);
int015(9, 64, [`StartRep`
1
]);
SetStatus(0);
int018;


{Atom: EW_ModelParameters}

sets;
AtomByName([EW_ModelParameters], Main);
if(not(AtomExists), Error([Cannot find mother atom 'EW_ModelParameters'. Inheriting from BaseClass.]));
CreateAtom(a, s, [EW_ModelParameters], 1, false);
int023([], 0, 3080);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(17);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: EW_Experiment}

sets;
AtomByName([EW_Experiment], Main);
if(not(AtomExists), Error([Cannot find mother atom 'EW_Experiment'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EW_Experiment], 1, false);
SetExprAtt([RunLength], [hr(24)]);
SetAtt([nruns], 5);
SetAtt([curRun], 0);
SetTextAtt([path], []);
SetAtt([isRunningAsExp], 0);
SetExprAtt([OnResetCode], [0]);
SetExprAtt([AfterRunCode], [0]);
SetAtt([StartRep], 0);
SetTextAtt([CurPath], []);
SetAtt([StartTime], 0);
SetAtt([MultiLogTimes], 0);
SetAtt([MultiLogs], 0);
int023([], 15, 527528);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
SetChannelRanges(0, 0, 0, 1);
int001(18);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;


{Atom: UserEvents}

sets;
AtomByName([UserEvents], Main);
if(not(AtomExists), Error([Cannot find mother atom 'UserEvents'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [UserEvents], 1, false);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffSetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
int023([], 0, 525514);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(19);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstTempAgents}

sets;
AtomByName([List], Main);
if(not(AtomExists), Error([Cannot find mother atom 'List'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstTempAgents], 1, false);
int023([], 0, 1024);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(20);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: CameraPositions}

sets;
AtomByName([CameraPositions], Main);
if(not(AtomExists), Error([Cannot find mother atom 'CameraPositions'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [CameraPositions], 1, false);
SetAtt([Default2D_Index], 1);
SetAtt([Default3D_Index], 1);
SetAtt([AtomToFollow], 0);
SetAtt([obj2D_CameraPositions], 0);
SetAtt([obj3D_CameraPositions], 0);
SetExprAtt([LoadComboBoxFunction], [CameraPositions_AddCameraPositionsToComboBox(MAINMENUFORM)]);
int023([], 0, 525514);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(21);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 15);
int015(0, 64, [`ID`
1
]);
int015(1, 64, [`Show`
]);
int015(2, 0, [`Name`
]);
int015(3, 0, [`2D3D`
]);
int015(4, 0, [`Visible Height layers`
]);
int015(5, 0, [`Window Width`
]);
int015(6, 0, [`Window Heigth`
]);
int015(7, 0, [`Window Xloc`
]);
int015(8, 0, [`Window Yloc`
]);
int015(9, 0, [`Scale`
]);
int015(10, 0, [`ViewX`
]);
int015(11, 0, [`ViewY`
]);
int015(12, 0, [`ViewZ`
]);
int015(13, 0, [`Roll`
]);
int015(14, 0, [`Pitch`
]);
int015(15, 0, [`Yaw`
]);
SetStatus(0);
int018;


{Atom: 2D_CameraPositions}

sets;
AtomByName([2D_CameraPositions], Main);
if(not(AtomExists), Error([Cannot find mother atom '2D_CameraPositions'. Inheriting from BaseClass.]));
CreateAtom(a, s, [2D_CameraPositions], 1, false);
int023([], 0, 525514);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(22);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 15);
int015(0, 64, [`ID`
1
]);
int015(1, 64, [`Show`
]);
int015(2, 0, [`Name`
]);
int015(3, 0, [`2D3D`
]);
int015(4, 0, [`Visible Height layers`
]);
int015(5, 0, [`Window Width`
]);
int015(6, 0, [`Window Heigth`
]);
int015(7, 0, [`Window Xloc`
]);
int015(8, 0, [`Window Yloc`
]);
int015(9, 0, [`Scale`
]);
int015(10, 0, [`ViewX`
]);
int015(11, 0, [`ViewY`
]);
int015(12, 0, [`ViewZ`
]);
int015(13, 0, [`Roll`
]);
int015(14, 0, [`Pitch`
]);
int015(15, 0, [`Yaw`
]);
SetStatus(0);
int018;


{Atom: 3D_CameraPositions}

sets;
AtomByName([3D_CameraPositions], Main);
if(not(AtomExists), Error([Cannot find mother atom '3D_CameraPositions'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [3D_CameraPositions], 1, false);
int023([], 0, 525514);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(23);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 15);
int015(0, 64, [`ID`
1
]);
int015(1, 64, [`Show`
]);
int015(2, 0, [`Name`
]);
int015(3, 0, [`2D3D`
]);
int015(4, 0, [`Visible Height layers`
]);
int015(5, 0, [`Window Width`
]);
int015(6, 0, [`Window Heigth`
]);
int015(7, 0, [`Window Xloc`
]);
int015(8, 0, [`Window Yloc`
]);
int015(9, 0, [`Scale`
]);
int015(10, 0, [`ViewX`
]);
int015(11, 0, [`ViewY`
]);
int015(12, 0, [`ViewZ`
]);
int015(13, 0, [`Roll`
]);
int015(14, 0, [`Pitch`
]);
int015(15, 0, [`Yaw`
]);
SetStatus(0);
int018;
Up;
Up;


{Atom: PD_Environment_DisplayAtom}

sets;
AtomByName([DisplayAtom], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DisplayAtom'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [PD_Environment_DisplayAtom], 1, false);
int023([], 0, 1065984);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\Tools.ico]));
Layer(LayerByName([DoNotDraw]));
int001(24);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: PD_Environment}

sets;
AtomByName([PD_Environment], Main);
if(not(AtomExists), Error([Cannot find mother atom 'PD_Environment'. Inheriting from BaseClass.]));
CreateAtom(a, s, [PD_Environment], 1, false);
SetAtt([minX], -1669.75);
SetAtt([minY], -1006.54);
SetAtt([maxX], 8911.4398);
SetAtt([maxY], 10066.5098);
SetAtt([NrPortals], 0);
SetAtt([ContainsECMNetwork], 0);
SetAtt([ActivityID], 116);
SetExprAtt([3DBackgroundColor], [ColorSkyBlue]);
SetExprAtt([2DBackgroundColor], [ColorWhite]);
SetAtt([GridSize], 0.01);
SetAtt([ActiveHeightLayer], 1);
SetTextAtt([ShowLocation], [0]);
SetAtt([DisableStitching], 0);
SetAtt([DisablePreProcessing], 0);
SetAtt([IndicativeCorridorID], 0);
SetAtt([ActivityIDDynamic], 0);
SetAtt([MinimumClearance], 0.1);
SetExprAtt([ECMInformationType], [ECMINFORMATIONTYPE_MEDIALAXIS]);
SetAtt([ShowExcludeFromNetwork], 1);
SetAtt([LayoutChecked], 0);
int023([], 0, 551984);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([PD_Environment]));
int001(25);
SetSize(10581.1898, 11073.0498, 0);
SetTranslation(-1669.75, -1006.54, 0);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(1, 3);
int015(0, 64, [0
1
]);
int015(1, 64, [`Layer`
]);
int015(2, 64, [`Rank`
]);
int015(3, 64, [`Ref`
]);
int037(4);
int038(0, -1659.75, -996.54, 0);
int038(1, 8901.4398, -996.54, 0);
int038(2, 8901.4398, 10056.5098, 0);
int038(3, -1659.75, 10056.5098, 0);
int039;
SetStatus(0);
int018;


{Atom: HeightLayer_1}

sets;
AtomByName([HeightLayer], Main);
if(not(AtomExists), Error([Cannot find mother atom 'HeightLayer'. Inheriting from BaseClass.]));
CreateAtom(a, s, [HeightLayer_1], 1, false);
SetAtt([CheckContentInView], 2);
SetAtt([xCenter2D], 3583.44365);
SetAtt([yCenter2D], 4486.3299);
SetAtt([xRadius2D], 5243.19365);
SetAtt([yRadius2D], 5482.8699);
SetAtt([xCenter3D], 3583.44365);
SetAtt([yCenter3D], 4486.3299);
SetAtt([zCenter3D], 0.5);
SetAtt([Radius3D], 7586.36554891249);
SetAtt([lstWalkableAreas], 0);
SetAtt([lstObstacles], 0);
SetAtt([lstOpenings], 0);
SetAtt([lstUserActions], 0);
SetAtt([********], 0);
SetAtt([lstEdges], 0);
SetAtt([lstAgents], 0);
SetAtt([lstActivities], 0);
SetAtt([LayerID], 1);
SetExprAtt([PrimitiveType], [PRIMITIVETYPE_RECTANGLE]);
SetAtt([GL_Type], 7);
SetAtt([Winding], 0);
SetAtt([ECMColor], 0);
SetAtt([AllowedPrimitiveTypes], 3);
SetAtt([DrawBackground2D], 0);
SetAtt([DrawBackground3D], 0);
SetAtt([BackgroundScale], 1);
SetAtt([BackgroundXLoc], -13500);
SetAtt([BackgroundYLoc], -4050);
SetAtt([BackgroundRotate], 0);
SetAtt([Transparency], 0);
SetAtt([DrawInterior], 2);
SetAtt([DrawExterior], 3);
SetAtt([TextureScaleX], 1);
SetAtt([TextureScaleY], 1);
SetAtt([TextureTranslateX], 0);
SetAtt([TextureTranslateY], 0);
SetAtt([ColorInterior2D], 16581135);
SetAtt([ColorExterior2D], 0);
SetAtt([ColorExterior3D], 0);
SetAtt([TextureID], 0);
SetAtt([**NOTUSED**], 0);
SetAtt([HeightDifference], 0);
SetAtt([DrawMultipleContours], 0);
SetAtt([Height], 0);
SetAtt([DrawSides], 4);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([RotationContent], 0);
SetAtt([**NOTUSED2**], 0);
SetAtt([ExcludeFromNetwork], 0);
SetAtt([DefaultVisibility], 3);
SetAtt([lstResources], 0);
SetAtt([lstTransportation], 0);
SetAtt([zRadius3D], 0.5);
SetAtt([AlwaysCreateWalkableArea], 0);
SetExprAtt([ResetCode], [0]);
SetAtt([BackgroundZLoc], 0);
SetAtt([UseBoundingBox], 0);
int023([], 8421504, 533552);
RegisterIconAndModel3D(ModDir([Madrid_Estaciones_Metro\estaciones_metro_mod.gml]), [estaciones_metro_mod.gml], 1, 1, 1, 0, 0, 0, 0, 0, -8.5, 2, 7, 0, 3);
Set(Icon(a), IconByName([estaciones_metro_mod.gml]));
RegisterIconAndModel3D(ModDir([Madrid_Estaciones_Metro\estaciones_metro_mod.gml]), [estaciones_metro_mod.gml], 1, 1, 1, 0, 0, 0, 0, 0, -8.5, 2, 7, 0, 3);
AddModel3D(Model3DByName([estaciones_metro_mod.gml]), a);
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([HeightLayers]));
SetChannelRanges(0, 0, 0, 999);
int001(26);
SetSize(1057.57, 833.23, 0);
SetTranslation(-1659.75, -996.54, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
int037(4);
int038(0, 0, 833.23, 0);
int038(1, 1057.57, 833.23, 0);
int038(2, 1057.57, 0, 0);
int038(3, 0, 0, 0);
int039;
Set(Points_PrimitiveType(a), 2);
SetStatus(0);
int018;


{Atom: lstWalkableAreas}

sets;
AtomByName([lstWalkableAreas], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstWalkableAreas'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstWalkableAreas], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(27);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstObstacles}

sets;
AtomByName([lstObstacles], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstObstacles'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstObstacles], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(28);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstOpenings}

sets;
AtomByName([lstOpenings], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstOpenings'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstOpenings], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(29);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstTransportation}

sets;
AtomByName([lstTransportation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstTransportation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstTransportation], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(30);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstActivities}

sets;
AtomByName([lstActivities], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstActivities'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstActivities], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(31);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: EntryExit_1}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, s, [EntryExit_1], 1, false);
SetAtt([ID], 1);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(76);
SetLoc(4430.7799, 552.62, 0);
SetSize(24.695, 24.695, 1);
SetTranslation(-12.3475, -12.3475, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 24.695, 0, 0);
int038(2, 24.695, 24.695, 0);
int038(3, 0, 24.695, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_2}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_2], 1, false);
SetAtt([ID], 2);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(77);
SetLoc(3652.2499, -146.6, 0);
SetSize(38.57, 38.57, 1);
SetTranslation(-19.285, -19.285, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 38.57, 0, 0);
int038(2, 38.57, 38.57, 0);
int038(3, 0, 38.57, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_3}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_3], 1, false);
SetAtt([ID], 3);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(78);
SetLoc(3047.8499, 559.55, 0);
SetSize(32.58, 32.58, 1);
SetTranslation(-16.29, -16.29, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 32.58, 0, 0);
int038(2, 32.58, 32.58, 0);
int038(3, 0, 32.58, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_4}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_4], 1, false);
SetAtt([ID], 4);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(79);
SetLoc(3636.7999, 1043.23, 0);
SetSize(30.04, 30.04, 1);
SetTranslation(-15.02, -15.02, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 30.04, 0, 0);
int038(2, 30.04, 30.04, 0);
int038(3, 0, 30.04, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_5}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_5], 1, false);
SetAtt([ID], 5);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(80);
SetLoc(3658.6399, 1592.54, 0);
SetSize(26.1, 26.1, 1);
SetTranslation(-13.05, -13.05, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 26.1, 0, 0);
int038(2, 26.1, 26.1, 0);
int038(3, 0, 26.1, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_6}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_6], 1, false);
SetAtt([ID], 6);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(81);
SetLoc(4796.8699, 1447.86, 0);
SetSize(17.51, 17.51, 1);
SetTranslation(-8.755, -8.755, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 17.51, 0, 0);
int038(2, 17.51, 17.51, 0);
int038(3, 0, 17.51, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_7}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_7], 1, false);
SetAtt([ID], 7);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(82);
SetLoc(6256.6799, 1059.85, 0);
SetSize(23.22, 23.22, 1);
SetTranslation(-11.61, -11.61, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 23.22, 0, 0);
int038(2, 23.22, 23.22, 0);
int038(3, 0, 23.22, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_8}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_8], 1, false);
SetAtt([ID], 8);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(83);
SetLoc(7055.9798, 1443.7, 0);
SetSize(27.84, 27.84, 1);
SetTranslation(-13.92, -13.92, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 27.84, 0, 0);
int038(2, 27.84, 27.84, 0);
int038(3, 0, 27.84, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_9}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_9], 1, false);
SetAtt([ID], 9);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(84);
SetLoc(7907.9198, 1034.86, 0);
SetSize(27.59, 27.59, 1);
SetTranslation(-13.795, -13.795, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 27.59, 0, 0);
int038(2, 27.59, 27.59, 0);
int038(3, 0, 27.59, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_10}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_10], 1, false);
SetAtt([ID], 10);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(85);
SetLoc(8243.8898, 1442.53, 0);
SetSize(26.22, 26.22, 1);
SetTranslation(-13.11, -13.11, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 26.22, 0, 0);
int038(2, 26.22, 26.22, 0);
int038(3, 0, 26.22, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_11}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_11], 1, false);
SetAtt([ID], 11);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(86);
SetLoc(6521.5099, 1919.02, 0);
SetSize(0, 0, 1);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_12}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_12], 1, false);
SetAtt([ID], 12);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(87);
SetLoc(5998.2899, 2591.3299, 0);
SetSize(24.28, 24.28, 1);
SetTranslation(-12.14, -12.14, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 24.28, 0, 0);
int038(2, 24.28, 24.28, 0);
int038(3, 0, 24.28, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_13}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_13], 1, false);
SetAtt([ID], 13);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(88);
SetLoc(6850.7748, 2479.0249, 0);
SetSize(22.01, 22.01, 1);
SetTranslation(-11.005, -11.005, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 22.01, 0, 0);
int038(2, 22.01, 22.01, 0);
int038(3, 0, 22.01, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_14}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_14], 1, false);
SetAtt([ID], 14);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(89);
SetLoc(7118.2423, 2973.6824, 0);
SetSize(16.585, 16.585, 1);
SetTranslation(-8.2925, -8.2925, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 16.585, 0, 0);
int038(2, 16.585, 16.585, 0);
int038(3, 0, 16.585, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_15}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_15], 1, false);
SetAtt([ID], 15);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(90);
SetLoc(5401.5199, 2297.0799, 0);
SetSize(24.52, 24.52, 1);
SetTranslation(-12.26, -12.26, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 24.52, 0, 0);
int038(2, 24.52, 24.52, 0);
int038(3, 0, 24.52, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_16}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_16], 1, false);
SetAtt([ID], 16);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(91);
SetLoc(3524.5399, 2217.8, 0);
SetSize(35.97, 35.97, 1);
SetTranslation(-17.985, -17.985, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 35.97, 0, 0);
int038(2, 35.97, 35.97, 0);
int038(3, 0, 35.97, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_17}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_17], 1, false);
SetAtt([ID], 18);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(92);
SetLoc(6555.9499, 1976.42, 0);
SetSize(68.88, 68.88, 1);
SetTranslation(-34.44, -34.44, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 68.88, 0, 0);
int038(2, 68.88, 68.88, 0);
int038(3, 0, 68.88, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_18}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_18], 1, false);
SetAtt([ID], 19);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(93);
SetLoc(5496.0699, 2852.1899, 0);
SetSize(12.45, 12.45, 1);
SetTranslation(-6.225, -6.225, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 12.45, 0, 0);
int038(2, 12.45, 12.45, 0);
int038(3, 0, 12.45, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_19}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_19], 1, false);
SetAtt([ID], 20);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(94);
SetLoc(3719.2699, 2849.4799, 0);
SetSize(31, 31, 1);
SetTranslation(-15.5, -15.5, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 31, 0, 0);
int038(2, 31, 31, 0);
int038(3, 0, 31, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_20}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_20], 1, false);
SetAtt([ID], 21);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(95);
SetLoc(4005.0599, 3724.5499, 0);
SetSize(19.75, 19.75, 1);
SetTranslation(-9.875, -9.875, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 19.75, 0, 0);
int038(2, 19.75, 19.75, 0);
int038(3, 0, 19.75, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_21}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_21], 1, false);
SetAtt([ID], 22);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(96);
SetLoc(5190.4399, 3307.8899, 0);
SetSize(15.64, 15.64, 1);
SetTranslation(-7.82, -7.82, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 15.64, 0, 0);
int038(2, 15.64, 15.64, 0);
int038(3, 0, 15.64, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_22}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_22], 1, false);
SetAtt([ID], 23);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(97);
SetLoc(5958.1249, 3540.6949, 0);
SetSize(15.75, 15.75, 1);
SetTranslation(-7.875, -7.875, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 15.75, 0, 0);
int038(2, 15.75, 15.75, 0);
int038(3, 0, 15.75, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_23}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_23], 1, false);
SetAtt([ID], 24);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(98);
SetLoc(7041.4098, 3559.2399, 0);
SetSize(22.35, 22.35, 1);
SetTranslation(-11.175, -11.175, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 22.35, 0, 0);
int038(2, 22.35, 22.35, 0);
int038(3, 0, 22.35, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_24}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_24], 1, false);
SetAtt([ID], 25);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(99);
SetLoc(7102.6898, 3705.1599, 0);
SetSize(18.26, 18.26, 1);
SetTranslation(-9.13, -9.13, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.26, 0, 0);
int038(2, 18.26, 18.26, 0);
int038(3, 0, 18.26, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_25}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_25], 1, false);
SetAtt([ID], 26);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(100);
SetLoc(8414.2498, 4067.2599, 0);
SetSize(35.93, 35.93, 1);
SetTranslation(-17.965, -17.965, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 35.93, 0, 0);
int038(2, 35.93, 35.93, 0);
int038(3, 0, 35.93, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_26}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_26], 1, false);
SetAtt([ID], 27);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(101);
SetLoc(8811.4898, 3801.6199, 0);
SetSize(30.295, 30.295, 1);
SetTranslation(-15.1475, -15.1475, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 30.295, 0, 0);
int038(2, 30.295, 30.295, 0);
int038(3, 0, 30.295, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_27}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_27], 1, false);
SetAtt([ID], 28);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(102);
SetLoc(7916.0498, 4670.6099, 0);
SetSize(29.715, 29.715, 1);
SetTranslation(-14.8575, -14.8575, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 29.715, 0, 0);
int038(2, 29.715, 29.715, 0);
int038(3, 0, 29.715, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_28}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_28], 1, false);
SetAtt([ID], 29);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(103);
SetLoc(8212.7898, 5672.4399, 0);
SetSize(27.87, 27.87, 1);
SetTranslation(-13.935, -13.935, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 27.87, 0, 0);
int038(2, 27.87, 27.87, 0);
int038(3, 0, 27.87, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_29}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_29], 1, false);
SetAtt([ID], 30);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(104);
SetLoc(8647.1598, 6409.6999, 0);
SetSize(23.95, 23.95, 1);
SetTranslation(-11.975, -11.975, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 23.95, 0, 0);
int038(2, 23.95, 23.95, 0);
int038(3, 0, 23.95, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_30}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_30], 1, false);
SetAtt([ID], 31);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(105);
SetLoc(8161.1948, 6610.4249, 0);
SetSize(24.83, 24.83, 1);
SetTranslation(-12.415, -12.415, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 24.83, 0, 0);
int038(2, 24.83, 24.83, 0);
int038(3, 0, 24.83, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_31}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_31], 1, false);
SetAtt([ID], 32);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(106);
SetLoc(7210.4248, 5161.8549, 0);
SetSize(31.59, 31.59, 1);
SetTranslation(-15.795, -15.795, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 31.59, 0, 0);
int038(2, 31.59, 31.59, 0);
int038(3, 0, 31.59, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_32}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_32], 1, false);
SetAtt([ID], 33);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(107);
SetLoc(7457.5898, 5680.3599, 0);
SetSize(26.33, 26.33, 1);
SetTranslation(-13.165, -13.165, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 26.33, 0, 0);
int038(2, 26.33, 26.33, 0);
int038(3, 0, 26.33, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_33}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_33], 1, false);
SetAtt([ID], 34);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(108);
SetLoc(7692.8298, 6952.5198, 0);
SetSize(24.62, 24.62, 1);
SetTranslation(-12.31, -12.31, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 24.62, 0, 0);
int038(2, 24.62, 24.62, 0);
int038(3, 0, 24.62, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_34}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_34], 1, false);
SetAtt([ID], 35);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(109);
SetLoc(7346.2898, 6382.7499, 0);
SetSize(17.73, 17.73, 1);
SetTranslation(-8.865, -8.865, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 17.73, 0, 0);
int038(2, 17.73, 17.73, 0);
int038(3, 0, 17.73, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_35}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_35], 1, false);
SetAtt([ID], 36);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(110);
SetLoc(7326.4298, 6486.9899, 0);
SetSize(16.36, 16.36, 1);
SetTranslation(-8.18, -8.18, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 16.36, 0, 0);
int038(2, 16.36, 16.36, 0);
int038(3, 0, 16.36, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_36}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_36], 1, false);
SetAtt([ID], 37);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(111);
SetLoc(7226.5898, 6197.1199, 0);
SetSize(24.86, 24.86, 1);
SetTranslation(-12.43, -12.43, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 24.86, 0, 0);
int038(2, 24.86, 24.86, 0);
int038(3, 0, 24.86, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_37}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_37], 1, false);
SetAtt([ID], 38);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(112);
SetLoc(4784.1699, 4001.4299, 0);
SetSize(18.795, 18.795, 1);
SetTranslation(-9.3975, -9.3975, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.795, 0, 0);
int038(2, 18.795, 18.795, 0);
int038(3, 0, 18.795, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_38}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_38], 1, false);
SetAtt([ID], 39);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(113);
SetLoc(5891.9249, 4260.0549, 0);
SetSize(28.99, 28.99, 1);
SetTranslation(-14.495, -14.495, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 28.99, 0, 0);
int038(2, 28.99, 28.99, 0);
int038(3, 0, 28.99, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_39}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_39], 1, false);
SetAtt([ID], 40);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(114);
SetLoc(4744.4799, 4366.6599, 0);
SetSize(18.1, 18.1, 1);
SetTranslation(-9.05, -9.05, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.1, 0, 0);
int038(2, 18.1, 18.1, 0);
int038(3, 0, 18.1, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_40}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_40], 1, false);
SetAtt([ID], 41);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(115);
SetLoc(4711.3799, 4817.9699, 0);
SetSize(18.2, 18.2, 1);
SetTranslation(-9.1, -9.1, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.2, 0, 0);
int038(2, 18.2, 18.2, 0);
int038(3, 0, 18.2, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_41}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_41], 1, false);
SetAtt([ID], 42);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(116);
SetLoc(5719.4499, 4825.0199, 0);
SetSize(26.04, 26.04, 1);
SetTranslation(-13.02, -13.02, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 26.04, 0, 0);
int038(2, 26.04, 26.04, 0);
int038(3, 0, 26.04, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_42}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_42], 1, false);
SetAtt([ID], 43);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(117);
SetLoc(6997.0898, 4261.8399, 0);
SetSize(15.78, 15.78, 1);
SetTranslation(-7.89, -7.89, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 15.78, 0, 0);
int038(2, 15.78, 15.78, 0);
int038(3, 0, 15.78, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_43}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_43], 1, false);
SetAtt([ID], 44);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(118);
SetLoc(6461.1899, 5160.8499, 0);
SetSize(20.97, 20.97, 1);
SetTranslation(-10.485, -10.485, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 20.97, 0, 0);
int038(2, 20.97, 20.97, 0);
int038(3, 0, 20.97, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_44}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_44], 1, false);
SetAtt([ID], 45);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(119);
SetLoc(6944.4798, 5033.7699, 0);
SetSize(29.84, 29.84, 1);
SetTranslation(-14.92, -14.92, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 29.84, 0, 0);
int038(2, 29.84, 29.84, 0);
int038(3, 0, 29.84, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_45}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_45], 1, false);
SetAtt([ID], 46);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(120);
SetLoc(7087.1798, 5837.6999, 0);
SetSize(19.23, 19.23, 1);
SetTranslation(-9.615, -9.615, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 19.23, 0, 0);
int038(2, 19.23, 19.23, 0);
int038(3, 0, 19.23, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_46}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_46], 1, false);
SetAtt([ID], 47);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(121);
SetLoc(6561.0099, 6378.6599, 0);
SetSize(17.98, 17.98, 1);
SetTranslation(-8.99, -8.99, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 17.98, 0, 0);
int038(2, 17.98, 17.98, 0);
int038(3, 0, 17.98, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_47}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_47], 1, false);
SetAtt([ID], 48);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(122);
SetLoc(6837.8098, 6723.9098, 0);
SetSize(16.98, 16.98, 1);
SetTranslation(-8.49, -8.49, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 16.98, 0, 0);
int038(2, 16.98, 16.98, 0);
int038(3, 0, 16.98, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_48}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_48], 1, false);
SetAtt([ID], 49);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(123);
SetLoc(7161.5298, 6763.3598, 0);
SetSize(20.82, 20.82, 1);
SetTranslation(-10.41, -10.41, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 20.82, 0, 0);
int038(2, 20.82, 20.82, 0);
int038(3, 0, 20.82, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_49}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_49], 1, false);
SetAtt([ID], 50);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(124);
SetLoc(7092.9698, 7339.9898, 0);
SetSize(18.68, 18.68, 1);
SetTranslation(-9.34, -9.34, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.68, 0, 0);
int038(2, 18.68, 18.68, 0);
int038(3, 0, 18.68, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_50}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_50], 1, false);
SetAtt([ID], 51);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(125);
SetLoc(7167.4898, 7318.9498, 0);
SetSize(17.05, 17.05, 1);
SetTranslation(-8.525, -8.525, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 17.05, 0, 0);
int038(2, 17.05, 17.05, 0);
int038(3, 0, 17.05, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_51}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_51], 1, false);
SetAtt([ID], 52);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(126);
SetLoc(7748.2798, 7551.8898, 0);
SetSize(15.97, 15.97, 1);
SetTranslation(-7.985, -7.985, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 15.97, 0, 0);
int038(2, 15.97, 15.97, 0);
int038(3, 0, 15.97, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_52}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_52], 1, false);
SetAtt([ID], 53);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(127);
SetLoc(7671.7098, 8378.7498, 0);
SetSize(28.96, 28.96, 1);
SetTranslation(-14.48, -14.48, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 28.96, 0, 0);
int038(2, 28.96, 28.96, 0);
int038(3, 0, 28.96, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_53}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_53], 1, false);
SetAtt([ID], 54);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(128);
SetLoc(8360.2898, 8826.4398, 0);
SetSize(26.87, 26.87, 1);
SetTranslation(-13.435, -13.435, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 26.87, 0, 0);
int038(2, 26.87, 26.87, 0);
int038(3, 0, 26.87, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_54}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_54], 1, false);
SetAtt([ID], 55);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(129);
SetLoc(6943.4798, 8061.3298, 0);
SetSize(19.95, 19.95, 1);
SetTranslation(-9.975, -9.975, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 19.95, 0, 0);
int038(2, 19.95, 19.95, 0);
int038(3, 0, 19.95, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_55}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_55], 1, false);
SetAtt([ID], 56);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(130);
SetLoc(6738.5998, 7562.4798, 0);
SetSize(18.29, 18.29, 1);
SetTranslation(-9.145, -9.145, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.29, 0, 0);
int038(2, 18.29, 18.29, 0);
int038(3, 0, 18.29, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_56}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_56], 1, false);
SetAtt([ID], 57);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(131);
SetLoc(6766.8598, 7314.6898, 0);
SetSize(19.67, 19.67, 1);
SetTranslation(-9.835, -9.835, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 19.67, 0, 0);
int038(2, 19.67, 19.67, 0);
int038(3, 0, 19.67, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_57}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_57], 1, false);
SetAtt([ID], 58);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(132);
SetLoc(6521.9699, 7299.3098, 0);
SetSize(19.04, 19.04, 1);
SetTranslation(-9.52, -9.52, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 19.04, 0, 0);
int038(2, 19.04, 19.04, 0);
int038(3, 0, 19.04, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_58}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_58], 1, false);
SetAtt([ID], 59);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(133);
SetLoc(6245.9899, 7234.2998, 0);
SetSize(68.88, 68.88, 1);
SetTranslation(-34.44, -34.44, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 68.88, 0, 0);
int038(2, 68.88, 68.88, 0);
int038(3, 0, 68.88, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_59}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_59], 1, false);
SetAtt([ID], 60);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(134);
SetLoc(5854.4049, 7224.9848, 0);
SetSize(21.73, 21.73, 1);
SetTranslation(-10.865, -10.865, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 21.73, 0, 0);
int038(2, 21.73, 21.73, 0);
int038(3, 0, 21.73, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_60}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_60], 1, false);
SetAtt([ID], 61);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(135);
SetLoc(5983.0949, 6314.9649, 0);
SetSize(26.45, 26.45, 1);
SetTranslation(-13.225, -13.225, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 26.45, 0, 0);
int038(2, 26.45, 26.45, 0);
int038(3, 0, 26.45, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_61}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_61], 1, false);
SetAtt([ID], 62);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(136);
SetLoc(5787.0849, 5753.6649, 0);
SetSize(23.55, 23.55, 1);
SetTranslation(-11.775, -11.775, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 23.55, 0, 0);
int038(2, 23.55, 23.55, 0);
int038(3, 0, 23.55, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_62}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_62], 1, false);
SetAtt([ID], 63);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(137);
SetLoc(5801.8799, 5891.3999, 0);
SetSize(23.3, 23.3, 1);
SetTranslation(-11.65, -11.65, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 23.3, 0, 0);
int038(2, 23.3, 23.3, 0);
int038(3, 0, 23.3, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_63}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_63], 1, false);
SetAtt([ID], 64);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(138);
SetLoc(5155.7999, 5810.4799, 0);
SetSize(20.97, 20.97, 1);
SetTranslation(-10.485, -10.485, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 20.97, 0, 0);
int038(2, 20.97, 20.97, 0);
int038(3, 0, 20.97, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_64}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_64], 1, false);
SetAtt([ID], 65);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(139);
SetLoc(4884.0699, 5408.1099, 0);
SetSize(24.86, 24.86, 1);
SetTranslation(-12.43, -12.43, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 24.86, 0, 0);
int038(2, 24.86, 24.86, 0);
int038(3, 0, 24.86, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_65}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_65], 1, false);
SetAtt([ID], 66);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(140);
SetLoc(4598.1299, 5717.9899, 0);
SetSize(19.49, 19.49, 1);
SetTranslation(-9.745, -9.745, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 19.49, 0, 0);
int038(2, 19.49, 19.49, 0);
int038(3, 0, 19.49, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_66}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_66], 1, false);
SetAtt([ID], 67);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(141);
SetLoc(4764.2499, 5773.7199, 0);
SetSize(20.49, 20.49, 1);
SetTranslation(-10.245, -10.245, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 20.49, 0, 0);
int038(2, 20.49, 20.49, 0);
int038(3, 0, 20.49, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_67}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_67], 1, false);
SetAtt([ID], 68);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(142);
SetLoc(3886.8299, 5690.2299, 0);
SetSize(57.4, 57.4, 1);
SetTranslation(-28.7, -28.7, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 57.4, 0, 0);
int038(2, 57.4, 57.4, 0);
int038(3, 0, 57.4, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_68}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_68], 1, false);
SetAtt([ID], 69);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(143);
SetLoc(3972.6599, 4830.0999, 0);
SetSize(13.14, 13.14, 1);
SetTranslation(-6.57, -6.57, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 13.14, 0, 0);
int038(2, 13.14, 13.14, 0);
int038(3, 0, 13.14, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_69}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_69], 1, false);
SetAtt([ID], 70);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(144);
SetLoc(3383.3449, 4819.3749, 0);
SetSize(14.75, 14.75, 1);
SetTranslation(-7.375, -7.375, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 14.75, 0, 0);
int038(2, 14.75, 14.75, 0);
int038(3, 0, 14.75, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_70}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_70], 1, false);
SetAtt([ID], 71);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(145);
SetLoc(2775.7999, 5148.2899, 0);
SetSize(25.86, 25.86, 1);
SetTranslation(-12.93, -12.93, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 25.86, 0, 0);
int038(2, 25.86, 25.86, 0);
int038(3, 0, 25.86, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_71}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_71], 1, false);
SetAtt([ID], 72);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(146);
SetLoc(3370.1099, 6121.3199, 0);
SetSize(26.97, 26.97, 1);
SetTranslation(-13.485, -13.485, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 26.97, 0, 0);
int038(2, 26.97, 26.97, 0);
int038(3, 0, 26.97, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_72}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_72], 1, false);
SetAtt([ID], 73);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(147);
SetLoc(3674.7149, 6690.2049, 0);
SetSize(16.91, 16.91, 1);
SetTranslation(-8.455, -8.455, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 16.91, 0, 0);
int038(2, 16.91, 16.91, 0);
int038(3, 0, 16.91, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_73}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_73], 1, false);
SetAtt([ID], 74);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(148);
SetLoc(5135.4299, 6179.5199, 0);
SetSize(21.86, 21.86, 1);
SetTranslation(-10.93, -10.93, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 21.86, 0, 0);
int038(2, 21.86, 21.86, 0);
int038(3, 0, 21.86, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_74}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_74], 1, false);
SetAtt([ID], 75);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(149);
SetLoc(4691.9699, 6342.3499, 0);
SetSize(18.61, 18.61, 1);
SetTranslation(-9.305, -9.305, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.61, 0, 0);
int038(2, 18.61, 18.61, 0);
int038(3, 0, 18.61, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_75}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_75], 1, false);
SetAtt([ID], 76);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(150);
SetLoc(6264.0099, 7811.4298, 0);
SetSize(28.34, 28.34, 1);
SetTranslation(-14.17, -14.17, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 28.34, 0, 0);
int038(2, 28.34, 28.34, 0);
int038(3, 0, 28.34, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_76}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_76], 1, false);
SetAtt([ID], 77);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(151);
SetLoc(5149.6399, 8686.5398, 0);
SetSize(57.4, 57.4, 1);
SetTranslation(-28.7, -28.7, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 57.4, 0, 0);
int038(2, 57.4, 57.4, 0);
int038(3, 0, 57.4, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_77}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_77], 1, false);
SetAtt([ID], 78);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(152);
SetLoc(4681.8499, 8737.9398, 0);
SetSize(18.98, 18.98, 1);
SetTranslation(-9.49, -9.49, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.98, 0, 0);
int038(2, 18.98, 18.98, 0);
int038(3, 0, 18.98, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_78}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_78], 1, false);
SetAtt([ID], 79);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(153);
SetLoc(4354.1374, 8803.8173, 0);
SetSize(14.965, 14.965, 1);
SetTranslation(-7.4825, -7.4825, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 14.965, 0, 0);
int038(2, 14.965, 14.965, 0);
int038(3, 0, 14.965, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_79}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_79], 1, false);
SetAtt([ID], 80);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(154);
SetLoc(2687.3899, 8527.9098, 0);
SetSize(21.38, 21.38, 1);
SetTranslation(-10.69, -10.69, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 21.38, 0, 0);
int038(2, 21.38, 21.38, 0);
int038(3, 0, 21.38, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_80}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_80], 1, false);
SetAtt([ID], 81);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(155);
SetLoc(2002.585, 8251.8048, 0);
SetSize(31.43, 31.43, 1);
SetTranslation(-15.715, -15.715, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 31.43, 0, 0);
int038(2, 31.43, 31.43, 0);
int038(3, 0, 31.43, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_81}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_81], 1, false);
SetAtt([ID], 82);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(156);
SetLoc(3331.5949, 7641.3648, 0);
SetSize(30.79, 30.79, 1);
SetTranslation(-15.395, -15.395, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 30.79, 0, 0);
int038(2, 30.79, 30.79, 0);
int038(3, 0, 30.79, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_82}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_82], 1, false);
SetAtt([ID], 83);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(157);
SetLoc(4050.6649, 7438.1248, 0);
SetSize(11.27, 11.27, 1);
SetTranslation(-5.635, -5.635, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 11.27, 0, 0);
int038(2, 11.27, 11.27, 0);
int038(3, 0, 11.27, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_83}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_83], 1, false);
SetAtt([ID], 84);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(158);
SetLoc(4262.6599, 7362.8998, 0);
SetSize(14.4, 14.4, 1);
SetTranslation(-7.2, -7.2, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 14.4, 0, 0);
int038(2, 14.4, 14.4, 0);
int038(3, 0, 14.4, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_84}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_84], 1, false);
SetAtt([ID], 85);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(159);
SetLoc(3925.1849, 7051.3148, 0);
SetSize(17.25, 17.25, 1);
SetTranslation(-8.625, -8.625, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 17.25, 0, 0);
int038(2, 17.25, 17.25, 0);
int038(3, 0, 17.25, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_85}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_85], 1, false);
SetAtt([ID], 86);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(160);
SetLoc(4402.9199, 7332.7198, 0);
SetSize(12.72, 12.72, 1);
SetTranslation(-6.36, -6.36, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 12.72, 0, 0);
int038(2, 12.72, 12.72, 0);
int038(3, 0, 12.72, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_86}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_86], 1, false);
SetAtt([ID], 87);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(161);
SetLoc(4350.7899, 7701.2798, 0);
SetSize(20.155, 20.155, 1);
SetTranslation(-10.0775, -10.0775, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 20.155, 0, 0);
int038(2, 20.155, 20.155, 0);
int038(3, 0, 20.155, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_87}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_87], 1, false);
SetAtt([ID], 88);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(162);
SetLoc(4252.1499, 8069.8998, 0);
SetSize(18.1, 18.1, 1);
SetTranslation(-9.05, -9.05, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.1, 0, 0);
int038(2, 18.1, 18.1, 0);
int038(3, 0, 18.1, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_88}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_88], 1, false);
SetAtt([ID], 89);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(163);
SetLoc(4557.3199, 7851.2598, 0);
SetSize(21.31, 21.31, 1);
SetTranslation(-10.655, -10.655, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 21.31, 0, 0);
int038(2, 21.31, 21.31, 0);
int038(3, 0, 21.31, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_89}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_89], 1, false);
SetAtt([ID], 90);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(164);
SetLoc(4776.3999, 8218.5798, 0);
SetSize(21.91, 21.91, 1);
SetTranslation(-10.955, -10.955, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 21.91, 0, 0);
int038(2, 21.91, 21.91, 0);
int038(3, 0, 21.91, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_90}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_90], 1, false);
SetAtt([ID], 91);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(165);
SetLoc(4916.3499, 7884.9798, 0);
SetSize(21.04, 21.04, 1);
SetTranslation(-10.52, -10.52, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 21.04, 0, 0);
int038(2, 21.04, 21.04, 0);
int038(3, 0, 21.04, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_91}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_91], 1, false);
SetAtt([ID], 92);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(166);
SetLoc(5174.9699, 8089.4398, 0);
SetSize(20.12, 20.12, 1);
SetTranslation(-10.06, -10.06, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 20.12, 0, 0);
int038(2, 20.12, 20.12, 0);
int038(3, 0, 20.12, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_92}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_92], 1, false);
SetAtt([ID], 93);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(167);
SetLoc(5489.0574, 7888.4973, 0);
SetSize(27.105, 27.105, 1);
SetTranslation(-13.5525, -13.5525, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 27.105, 0, 0);
int038(2, 27.105, 27.105, 0);
int038(3, 0, 27.105, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_93}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_93], 1, false);
SetAtt([ID], 94);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(168);
SetLoc(5290.6199, 7451.5098, 0);
SetSize(11.2, 11.2, 1);
SetTranslation(-5.6, -5.6, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 11.2, 0, 0);
int038(2, 11.2, 11.2, 0);
int038(3, 0, 11.2, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_94}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_94], 1, false);
SetAtt([ID], 95);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(169);
SetLoc(5433.0799, 6988.1498, 0);
SetSize(23.06, 23.06, 1);
SetTranslation(-11.53, -11.53, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 23.06, 0, 0);
int038(2, 23.06, 23.06, 0);
int038(3, 0, 23.06, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_95}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_95], 1, false);
SetAtt([ID], 96);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(170);
SetLoc(5500.7799, 6871.3198, 0);
SetSize(16.48, 16.48, 1);
SetTranslation(-8.24, -8.24, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 16.48, 0, 0);
int038(2, 16.48, 16.48, 0);
int038(3, 0, 16.48, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_96}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_96], 1, false);
SetAtt([ID], 97);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(171);
SetLoc(5372.0799, 6862.4498, 0);
SetSize(13.42, 13.42, 1);
SetTranslation(-6.71, -6.71, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 13.42, 0, 0);
int038(2, 13.42, 13.42, 0);
int038(3, 0, 13.42, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_97}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_97], 1, false);
SetAtt([ID], 98);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(172);
SetLoc(4989.1724, 6829.3073, 0);
SetSize(10.115, 10.115, 1);
SetTranslation(-5.0575, -5.0575, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 10.115, 0, 0);
int038(2, 10.115, 10.115, 0);
int038(3, 0, 10.115, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_98}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_98], 1, false);
SetAtt([ID], 99);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(173);
SetLoc(4909.5699, 6814.3498, 0);
SetSize(12.13, 12.13, 1);
SetTranslation(-6.065, -6.065, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 12.13, 0, 0);
int038(2, 12.13, 12.13, 0);
int038(3, 0, 12.13, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_99}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_99], 1, false);
SetAtt([ID], 100);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(174);
SetLoc(4568.0799, 6722.9698, 0);
SetSize(12.64, 12.64, 1);
SetTranslation(-6.32, -6.32, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 12.64, 0, 0);
int038(2, 12.64, 12.64, 0);
int038(3, 0, 12.64, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_100}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_100], 1, false);
SetAtt([ID], 101);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(175);
SetLoc(4623.2074, 6819.6073, 0);
SetSize(14.905, 14.905, 1);
SetTranslation(-7.4525, -7.4525, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 14.905, 0, 0);
int038(2, 14.905, 14.905, 0);
int038(3, 0, 14.905, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_101}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_101], 1, false);
SetAtt([ID], 102);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(176);
SetLoc(4938.3499, 7187.5998, 0);
SetSize(14.22, 14.22, 1);
SetTranslation(-7.11, -7.11, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 14.22, 0, 0);
int038(2, 14.22, 14.22, 0);
int038(3, 0, 14.22, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_102}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_102], 1, false);
SetAtt([ID], 103);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(177);
SetLoc(4998.1599, 7153.4998, 0);
SetSize(16.3, 16.3, 1);
SetTranslation(-8.15, -8.15, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 16.3, 0, 0);
int038(2, 16.3, 16.3, 0);
int038(3, 0, 16.3, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_103}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_103], 1, false);
SetAtt([ID], 104);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(178);
SetLoc(1755.16, 9014.1898, 0);
SetSize(13.27, 13.27, 1);
SetTranslation(-6.635, -6.635, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 13.27, 0, 0);
int038(2, 13.27, 13.27, 0);
int038(3, 0, 13.27, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_104}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_104], 1, false);
SetAtt([ID], 105);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(179);
SetLoc(4140.5799, 9327.0198, 0);
SetSize(30.94, 30.94, 1);
SetTranslation(-15.47, -15.47, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 30.94, 0, 0);
int038(2, 30.94, 30.94, 0);
int038(3, 0, 30.94, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_105}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_105], 1, false);
SetAtt([ID], 106);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(180);
SetLoc(4103.5149, 9803.3548, 0);
SetSize(34.21, 34.21, 1);
SetTranslation(-17.105, -17.105, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 34.21, 0, 0);
int038(2, 34.21, 34.21, 0);
int038(3, 0, 34.21, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_106}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_106], 1, false);
SetAtt([ID], 107);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(181);
SetLoc(4487.5499, 9684.4398, 0);
SetSize(24.99, 24.99, 1);
SetTranslation(-12.495, -12.495, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 24.99, 0, 0);
int038(2, 24.99, 24.99, 0);
int038(3, 0, 24.99, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_107}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_107], 1, false);
SetAtt([ID], 108);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(182);
SetLoc(4843.5499, 9533.6398, 0);
SetSize(25.76, 25.76, 1);
SetTranslation(-12.88, -12.88, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 25.76, 0, 0);
int038(2, 25.76, 25.76, 0);
int038(3, 0, 25.76, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_108}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_108], 1, false);
SetAtt([ID], 109);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(183);
SetLoc(4955.2599, 9119.6898, 0);
SetSize(13.97, 13.97, 1);
SetTranslation(-6.985, -6.985, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 13.97, 0, 0);
int038(2, 13.97, 13.97, 0);
int038(3, 0, 13.97, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_109}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_109], 1, false);
SetAtt([ID], 110);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(184);
SetLoc(5704.6399, 9059.2898, 0);
SetSize(17.28, 17.28, 1);
SetTranslation(-8.64, -8.64, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 17.28, 0, 0);
int038(2, 17.28, 17.28, 0);
int038(3, 0, 17.28, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_110}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_110], 1, false);
SetAtt([ID], 111);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(185);
SetLoc(6032.9499, 9340.4498, 0);
SetSize(17.34, 17.34, 1);
SetTranslation(-8.67, -8.67, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 17.34, 0, 0);
int038(2, 17.34, 17.34, 0);
int038(3, 0, 17.34, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_111}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_111], 1, false);
SetAtt([ID], 112);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(186);
SetLoc(7615.0398, 9328.1798, 0);
SetSize(18.67, 18.67, 1);
SetTranslation(-9.335, -9.335, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 18.67, 0, 0);
int038(2, 18.67, 18.67, 0);
int038(3, 0, 18.67, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_112}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_112], 1, false);
SetAtt([ID], 113);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(187);
SetLoc(7369.1298, 9765.6898, 0);
SetSize(27.22, 27.22, 1);
SetTranslation(-13.61, -13.61, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 27.22, 0, 0);
int038(2, 27.22, 27.22, 0);
int038(3, 0, 27.22, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_113}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_113], 1, false);
SetAtt([ID], 114);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(188);
SetLoc(7233.1498, 9958.4298, 0);
SetSize(21.54, 21.54, 1);
SetTranslation(-10.77, -10.77, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 21.54, 0, 0);
int038(2, 21.54, 21.54, 0);
int038(3, 0, 21.54, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_114}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_114], 1, false);
SetAtt([ID], 115);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(189);
SetLoc(6745.5898, 9539.6398, 0);
SetSize(22.9999, 22.9999, 1);
SetTranslation(-11.49995, -11.49995, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 22.9999, 0, 0);
int038(2, 22.9999, 22.9999, 0);
int038(3, 0, 22.9999, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;


{Atom: EntryExit_115}

sets;
AtomByName([ActivityLocation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityLocation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [EntryExit_115], 1, false);
SetAtt([ID], 116);
SetExprAtt([ActivityType], [ACTIVITYTYPE_ENTRY_EXIT]);
SetAtt([PrimitiveType], 1);
SetExprAtt([AllowedPrimitiveTypes], [19 {with polygon and trapezoid: 31}]);
SetAtt([Radius], 0);
SetAtt([GL_Type], 7);
SetAtt([Winding], 1);
SetTextAtt([Group], [0]);
SetAtt([ServerQueues], 0);
SetAtt([ShowName], 1);
SetExprAtt([RepetitiveMode], [GENERATORTYPE_NOREPEAT]);
SetAtt([RepeatTime], 0);
SetAtt([OffsetTime], 0);
SetAtt([NrTimesToRepeat], 0);
SetAtt([RepeatCounter], 1);
SetExprAtt([LocationFrom], [{**Random**} Primitives_RandomPoint(c, Agent_GetRadius(i), PDEnvAtom)]);
SetExprAtt([LocationTo], [{**Continue from current location**} Points_Get(i, 1) ]);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([Available], 1);
SetAtt([DrawInterior], 3);
SetAtt([DrawExterior], 3);
SetExprAtt([ColorInterior2D], [Color(c)]);
SetExprAtt([ColorExterior2D], [4DS[GetPreference([Col2DFixedBBox])]4DS]);
SetAtt([ColorExterior3D], 0);
SetAtt([WaitingForQueue], 0);
SetAtt([ServiceTime], 0);
SetAtt([2DModelType], 0);
SetAtt([3DModelType], 0);
SetAtt([DrawingType], 0);
SetAtt([DrawingRadius], 1);
SetAtt([DrawingDirection], 1);
SetExprAtt([ActivityTime], [0]);
SetAtt([ServerQueueCapacity], 100000);
SetAtt([ExcludeFromNetwork], 13001);
SetAtt([KeepAgentsArray], 0);
SetAtt([QueueStrategy], 1);
SetAtt([QueueDiscipline], 0);
SetAtt([EnsureWalkableArea], 0);
SetExprAtt([AgentViewAngle], [v {**current rotation**}]);
SetAtt([LocationDeviation], 0);
SetAtt([TravelTime], 0);
SetAtt([TravelTimeColor], 0);
SetAtt([TiltEdge], 0);
SetAtt([TiltEdgeOpposite], 0);
SetAtt([TiltHeight], 0);
SetAtt([Slope], 0);
SetAtt([NrAgentsWaitingForTrigger], 0);
SetAtt([NumberOfAgents], 0);
SetAtt([TextSize], 0.75);
SetAtt([LineDestination], 0);
SetAtt([AvgServiceTime], 0);
SetExprAtt([ActivityTimeAfter**], [0]);
SetAtt([HighLevelRoutingPoint], 0);
SetAtt([ActivityNetworkNodeID], 0);
SetExprAtt([Direction], [DIRECTION_BIDIRECTIONAL]);
SetExprAtt([ResetCode], [0]);
SetExprAtt([UseIndicativeCorridor], [0]);
SetExprAtt([NextAgentStrategy], [4DS[{**Trigger first waiting agents in connected corridor**}do(  var([valIndex], vbValue, 1 {indicative corridor index}),  if(    IcConnected(ActivityLocation_GetServerQueues(c) + valIndex, c),    IndicativeCorridor_TriggerAllWaitingAgents(in(ActivityLocation_GetServerQueues(c) + valIndex, c), TRUE {condition}, c, 1 {nr to trigger})  ))]4DS]);
SetTextAtt([SidesBlocked], []);
SetExprAtt([StartRoutingTrigger], [0]);
int023([], 16711680, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([Activities]));
SetChannelRanges(0, 0, 0, 999);
int001(190);
SetLoc(5588.9199, 9795.2098, 0);
SetSize(26.69, 26.69, 1);
SetTranslation(-13.345, -13.345, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
SetTable(1, 4);
int015(0, 64, [``
1
]);
int015(1, 64, [`Start time`
-1
]);
int015(2, 76, [`Time open`
0
]);
int015(3, 93, [`Trigger on start`
0
]);
int015(4, 115, [`Trigger on end`
0
]);
int037(4);
int038(0, 0, 0, 0);
int038(1, 26.69, 0, 0);
int038(2, 26.69, 26.69, 0);
int038(3, 0, 26.69, 0);
int039;
Set(Points_PrimitiveType(a), 7);
SetStatus(11);
int018;
Up;


{Atom: lstUserActions}

sets;
AtomByName([lstUserActions], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstUserActions'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstUserActions], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(32);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstAgents}

sets;
AtomByName([lstAgents], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstAgents'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstAgents], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(33);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstResources}

sets;
AtomByName([lstResources], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstResources'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstResources], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(34);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: ECM_Visualization}

sets;
AtomByName([ECM_Visualization], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ECM_Visualization'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ECM_Visualization], 1, false);
int023([], 0, 542768);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([ECMNetwork]));
int001(35);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;


{Atom: HeightLayer_2}

sets;
AtomByName([HeightLayer], Main);
if(not(AtomExists), Error([Cannot find mother atom 'HeightLayer'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [HeightLayer_2], 1, false);
SetAtt([CheckContentInView], 2);
SetAtt([xCenter2D], 4450.7199);
SetAtt([yCenter2D], 4803.6599);
SetAtt([xRadius2D], 4450.7199);
SetAtt([yRadius2D], 5252.8499);
SetAtt([xCenter3D], 4450.7199);
SetAtt([yCenter3D], 4803.6599);
SetAtt([zCenter3D], 0.5);
SetAtt([Radius3D], 6884.86310322769);
SetAtt([lstWalkableAreas], 0);
SetAtt([lstObstacles], 0);
SetAtt([lstOpenings], 0);
SetAtt([lstUserActions], 0);
SetAtt([********], 0);
SetAtt([lstEdges], 0);
SetAtt([lstAgents], 0);
SetAtt([lstActivities], 0);
SetAtt([LayerID], 2);
SetAtt([PrimitiveType], 2);
SetAtt([GL_Type], 7);
SetAtt([Winding], 0);
SetAtt([ECMColor], 0);
SetAtt([AllowedPrimitiveTypes], 3);
SetAtt([DrawBackground2D], 2);
SetAtt([DrawBackground3D], 0);
SetAtt([BackgroundScale], 1);
SetAtt([BackgroundXLoc], 0);
SetAtt([BackgroundYLoc], 0);
SetAtt([BackgroundRotate], 0);
SetAtt([Transparency], 0);
SetAtt([DrawInterior], 2);
SetAtt([DrawExterior], 3);
SetAtt([TextureScaleX], 1);
SetAtt([TextureScaleY], 1);
SetAtt([TextureTranslateX], 0);
SetAtt([TextureTranslateY], 0);
SetAtt([ColorInterior2D], 8421504);
SetAtt([ColorExterior2D], 0);
SetAtt([ColorExterior3D], 0);
SetAtt([TextureID], 0);
SetAtt([**NOTUSED**], 0);
SetAtt([HeightDifference], 0);
SetAtt([DrawMultipleContours], 0);
SetAtt([Height], 0);
SetAtt([DrawSides], 4);
SetExprAtt([EntryTrigger], [0]);
SetExprAtt([ExitTrigger], [0]);
SetAtt([RotationContent], 0);
SetAtt([**NOTUSED2**], 0);
SetAtt([ExcludeFromNetwork], 0);
SetAtt([DefaultVisibility], 3);
SetAtt([lstResources], 0);
SetAtt([lstTransportation], 0);
SetAtt([zRadius3D], 0.5);
SetAtt([AlwaysCreateWalkableArea], 0);
SetExprAtt([ResetCode], [0]);
SetAtt([BackgroundZLoc], 0);
SetAtt([UseBoundingBox], 0);
int023([], 8421504, 533552);
RegisterIconAndModel3D(ModDir([Madrid_Hospitals_kml\madrid_hospitals_mod.gml]), [madrid_hospitals_mod.gml], 1, 1, 1, 0, 0, 0, 0, 0, -8.5, 2, 7, 0, 3);
Set(Icon(a), IconByName([madrid_hospitals_mod.gml]));
RegisterIconAndModel3D(ModDir([Madrid_Hospitals_kml\madrid_hospitals_mod.gml]), [madrid_hospitals_mod.gml], 1, 1, 1, 0, 0, 0, 0, 0, -8.5, 2, 7, 0, 3);
AddModel3D(Model3DByName([madrid_hospitals_mod.gml]), a);
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([HeightLayers]));
SetChannelRanges(0, 0, 0, 999);
int001(36);
SetSize(7446.5898, 10505.6998, 0);
SetTranslation(1454.85, -449.19, 0);
LockPosition(false);
LockSize(true);
DisableIconRotation(false);
int037(4);
int038(0, 0, 10505.6998, 0);
int038(1, 7446.5898, 10505.6998, 0);
int038(2, 7446.5898, 0, 0);
int038(3, 0, 0, 0);
int039;
Set(Points_PrimitiveType(a), 2);
SetStatus(0);
int018;


{Atom: lstWalkableAreas}

sets;
AtomByName([lstWalkableAreas], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstWalkableAreas'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstWalkableAreas], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(37);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstObstacles}

sets;
AtomByName([lstObstacles], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstObstacles'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstObstacles], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(38);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstOpenings}

sets;
AtomByName([lstOpenings], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstOpenings'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstOpenings], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(39);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstTransportation}

sets;
AtomByName([lstTransportation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstTransportation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstTransportation], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(40);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstActivities}

sets;
AtomByName([lstActivities], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstActivities'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstActivities], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(41);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstUserActions}

sets;
AtomByName([lstUserActions], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstUserActions'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstUserActions], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(42);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstAgents}

sets;
AtomByName([lstAgents], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstAgents'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstAgents], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(43);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstResources}

sets;
AtomByName([lstResources], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstResources'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstResources], 1, false);
int023([], 0, 527360);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(44);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: ECM_Visualization}

sets;
AtomByName([ECM_Visualization], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ECM_Visualization'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ECM_Visualization], 1, false);
int023([], 0, 542768);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([ECMNetwork]));
int001(45);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;
Up;


{Atom: AgentDrawInformation}

sets;
AtomByName([AgentDrawInformation], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentDrawInformation'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [AgentDrawInformation], 1, false);
SetAtt([InformationType2D], 14);
SetAtt([InformationType3D], 0);
int023([], 0, 546864);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([PD_Environment]));
int001(46);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: RunClock}

sets;
AtomByName([RunClock], Main);
if(not(AtomExists), Error([Cannot find mother atom 'RunClock'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [RunClock], 1, false);
SetExprAtt([text], [4DS[DateTime(TimeFromNow(att([startTime], c)), [hh:mm:ss])]4DS]);
SetExprAtt([textcolor], [1]);
SetExprAtt([backgroundcolor], [colorwhite]);
SetAtt([fontsize], 3);
SetExprAtt([font], [4DS[[Calibri]]4DS]);
SetAtt([fontscale], 0);
SetAtt([italic], 0);
SetAtt([pitch], 0);
SetAtt([xpitch], -45);
SetAtt([zpitch], 0);
SetExprAtt([3DEnabled], [Time > 0]);
SetAtt([FontSize3D], 0.2);
SetExprAtt([2DEnabled], [Time > 0]);
SetAtt([StartTime], 0);
SetAtt([EvacuationTime], 0);
SetAtt([OffSet3D], 0);
SetAtt([OffSet2D], 0);
int023([], 15, 542768);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\Tools.ico]));
Layer(LayerByName([PD_Environment]));
SetChannels(1, 0);
SetChannelRanges(1, 1, 0, 0);
int001(47);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: MovieRecord}

sets;
AtomByName([MovieRecord], Main);
if(not(AtomExists), Error([Cannot find mother atom 'MovieRecord'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [MovieRecord], 1, false);
SetExprAtt([Filename], [4DS[WorkDir(Concat(ExtractPreName(TextAtt(1, Model)), [.avi]))]4DS]);
SetAtt([Framerate], 5);
SetAtt([ActiveRecord], -1);
SetAtt([RecordPaused], 0);
SetAtt([Interval], 0.2);
SetAtt([FromGuiInstance], 0);
SetAtt([Width], 1600);
SetAtt([Height], 900);
SetExprAtt([RecordTime], [0]);
SetAtt([StopRecording], 1);
int023([], 15, 542720);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\Tools.ico]));
Layer(LayerByName([DoNotDraw]));
SetChannels(1, 0);
SetChannelRanges(1, 1, 0, 0);
int001(48);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: SelectionPolygon}

sets;
AtomByName([SelectionPolygon], Main);
if(not(AtomExists), Error([Cannot find mother atom 'SelectionPolygon'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [SelectionPolygon], 1, false);
SetAtt([PrimitiveType], 8);
SetAtt([GL_Type], 0);
SetAtt([Winding], 0);
int023([], 0, 528432);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(49);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
Set(Points_PrimitiveType(a), 10);
SetStatus(0);
int018;
Up;


{Atom: PD_Output}

sets;
AtomByName([PD_Output], Main);
if(not(AtomExists), Error([Cannot find mother atom 'PD_Output'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [PD_Output], 1, false);
SetExprAtt([StartTime], [0]);
SetExprAtt([EndTime], [0]);
SetAtt([ColorMapType], 0);
SetAtt([ShowLegend], 1);
SetAtt([LegendSizeX], 2.5);
SetAtt([ActivityRouteHide], 0);
SetAtt([OutputLoaded], 0);
SetAtt([CreateHeightMap], 0);
SetAtt([ColormapRef], 0);
SetTextAtt([ColormapTitle], [0]);
SetAtt([GridSize], 0.25);
SetAtt([ScaleLegend], 1);
SetAtt([TextSize], 1);
SetAtt([GridDensityRadius], 0.6);
SetAtt([SocialCostValue], 8.38);
SetTextAtt([SocialCostCurrency], [€]);
SetAtt([SocialCostDensityFactor], 0.6667);
SetAtt([SocialCostDensityMin], 0.5);
SetAtt([SocialCostDensityMax], 2);
SetAtt([SocialCostSet], 0);
int041(0, 12, [`Walking`;2;0.5;
`Walking`;2;0.5;
`Walking`;2;0.5;
`Stairs up`;4;0.5;
`Stairs down`;2.5;0.5;
`Escalator`;1.5;0;
`Escalator`;1.5;0;
`Stairs up`;4;0.5;
`Stairs down`;2.5;0.5;
`Stands`;2;0.5;
`Stairs up`;4;0.5;
`Stairs down`;2.5;0.5;
]);
int042(0, 0);
int042(1, 0);
int023([], 13158600, 541728);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([PD_Environment]));
int001(50);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: OutputLayer_1}

sets;
AtomByName([OutputLayer], Main);
if(not(AtomExists), Error([Cannot find mother atom 'OutputLayer'. Inheriting from BaseClass.]));
CreateAtom(a, s, [OutputLayer_1], 1, false);
SetAtt([lstFlowCounters], 0);
SetAtt([lstDensityAreas], 0);
SetAtt([lstColorMaps], 0);
SetAtt([LayerID], 1);
int023([], 8421504, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([HeightLayers]));
SetChannels(1, 0);
SetChannelRanges(1, 255, 0, 0);
int001(51);
SetSize(1057.57, 833.23, 0);
SetTranslation(-1659.75, -996.54, 0);
LockPosition(true);
LockSize(true);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstColorMaps}

sets;
AtomByName([lstColorMaps], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstColorMaps'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstColorMaps], 1, false);
int023([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([HeightLayers]));
int001(52);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstFlowCounters}

sets;
AtomByName([lstFlowCounters], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstFlowCounters'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstFlowCounters], 1, false);
int023([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([HeightLayers]));
int001(53);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstDensityAreas}

sets;
AtomByName([lstDensityAreas], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstDensityAreas'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstDensityAreas], 1, false);
int023([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([HeightLayers]));
int001(54);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstColorMaps}

sets;
AtomByName([lstColorMaps], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstColorMaps'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstColorMaps], 1, false);
int023([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([HeightLayers]));
int001(55);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;


{Atom: OutputLayer_2}

sets;
AtomByName([OutputLayer], Main);
if(not(AtomExists), Error([Cannot find mother atom 'OutputLayer'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [OutputLayer_2], 1, false);
SetAtt([lstFlowCounters], 0);
SetAtt([lstDensityAreas], 0);
SetAtt([lstColorMaps], 0);
SetAtt([LayerID], 2);
int023([], 8421504, 533552);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([HeightLayers]));
SetChannels(1, 0);
SetChannelRanges(1, 255, 0, 0);
int001(56);
SetSize(7446.5898, 10505.6998, 0);
SetTranslation(1454.85, -449.19, 0);
LockPosition(true);
LockSize(true);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstColorMaps}

sets;
AtomByName([lstColorMaps], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstColorMaps'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstColorMaps], 1, false);
int023([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
int001(57);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstFlowCounters}

sets;
AtomByName([lstFlowCounters], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstFlowCounters'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstFlowCounters], 1, false);
int023([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(58);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstDensityAreas}

sets;
AtomByName([lstDensityAreas], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstDensityAreas'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstDensityAreas], 1, false);
int023([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(59);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstColorMaps}

sets;
AtomByName([lstColorMaps], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstColorMaps'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [lstColorMaps], 1, false);
int023([], 0, 525312);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(60);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;


{Atom: OutputSettings}

sets;
AtomByName([List], Main);
if(not(AtomExists), Error([Cannot find mother atom 'List'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [OutputSettings], 1, false);
int023([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(61);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstNorms}

sets;
AtomByName([List], Main);
if(not(AtomExists), Error([Cannot find mother atom 'List'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstNorms], 1, false);
int023([], 0, 1024);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(62);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: DensityNorms}

sets;
AtomByName([DensityNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DensityNorms'. Inheriting from BaseClass.]));
CreateAtom(a, s, [DensityNorms], 1, false);
SetAtt([BlendColors], 0);
SetAtt([Interval], 30);
SetAtt([IntervalBetweenLabels], 0);
SetAtt([Threshold], 0);
SetAtt([BottomLabelsAngle], 0);
SetAtt([StatisticType], 16001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
SetAtt([MaxOfMeanIntervals], 1);
SetAtt([AutoRescaleNorms], 17002);
SetExprAtt([TimeAboveLowerBound], [{**>= Level E**}
CellAsValue(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([TimeAboveUpperBound], [{**No upper bound**}
0]);
SetAtt([UserDefinedColormaps], 0);
int023([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(63);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
int015(0, 64, [0
1
2
3
4
5
6
7
]);
int015(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
`F`
]);
int015(2, 98, [`Lower bound`
0
1E-6
0.308
0.431
0.718
1.076
2.153
]);
int015(3, 102, [`Upper bound`
1E-6
0.308
0.431
0.718
1.076
2.153
3
]);
int015(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
int015(5, 105, [`Description`
`empty`
`few`
`few/medium`
`medium`
`busy`
`very busy`
`congested`
]);
SetStatus(0);
int018;


{Atom: DensityNorms_Walkways}

sets;
AtomByName([DensityNorms_Walkways], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DensityNorms_Walkways'. Inheriting from BaseClass.]));
CreateAtom(a, s, [DensityNorms_Walkways], 1, false);
int023([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(64);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
int015(0, 64, [0
1
2
3
4
5
6
7
]);
int015(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
`F`
]);
int015(2, 98, [`Lower bound`
0
1E-6
0.308
0.431
0.718
1.076
2.153
]);
int015(3, 102, [`Upper bound`
1E-6
0.308
0.431
0.718
1.076
2.153
3
]);
int015(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
int015(5, 105, [`Description`
`empty`
`few`
`few/medium`
`medium`
`busy`
`very busy`
`congested`
]);
SetStatus(0);
int018;


{Atom: DensityNorms_Stairways}

sets;
AtomByName([DensityNorms_Stairways], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DensityNorms_Stairways'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [DensityNorms_Stairways], 1, false);
int023([], 0, 0);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(65);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
int015(0, 64, [0
1
2
3
4
5
6
7
]);
int015(1, 64, [`LOS`
`-`
`A`
`B`
`C`
`D`
`E`
`F`
]);
int015(2, 98, [`Lower bound`
0
1E-6
0.538
0.718
1.076
1.538
2.691
]);
int015(3, 102, [`Upper bound`
1E-6
0.538
0.718
1.076
1.538
2.691
3
]);
int015(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
int015(5, 105, [`Description`
`empty`
`few`
`few/medium`
`medium`
`busy`
`very busy`
`congested`
]);
SetStatus(0);
int018;


{Atom: DensityTimeNorms}

sets;
AtomByName([DensityTimeNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'DensityTimeNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [DensityTimeNorms], 1, false);
int023([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(66);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
int015(0, 64, [0
1
2
3
4
5
6
7
]);
int015(1, 64, [`Level`
0
1
2
3
4
5
6
]);
int015(2, 98, [`Lower bound`
0
1E-6
60
120
180
240
300
]);
int015(3, 102, [`Upper bound`
1E-6
60
120
180
240
300
301
]);
int015(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
int015(5, 64, [`Description`
``
`A`
`B`
`C`
`D`
`E`
`F`
]);
SetStatus(0);
int018;
Up;


{Atom: FrequencyNorms}

sets;
AtomByName([FrequencyNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'FrequencyNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [FrequencyNorms], 1, false);
SetAtt([AutoRescaleNorms], 17002);
SetAtt([BlendColors], 0);
SetAtt([Interval], 60);
SetAtt([IntervalsBetweenLabels], 0);
SetAtt([BottomLabelsAngle], 0);
SetExprAtt([DefaultGraphType], [GraphBar]);
SetAtt([DefaultGraphView3D], 0);
SetAtt([MinY], 0);
SetAtt([MaxY], 0);
SetAtt([ShowPerMeter], 0);
SetAtt([UserDefinedColormaps], 0);
SetAtt([StatisticType], 18001);
SetAtt([ShowOnLayer], 1);
SetAtt([CreateHeightMap], 0);
int023([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(67);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
int015(0, 64, [0
1
2
3
4
5
6
7
]);
int015(1, 64, [`Level`
0
1
2
3
4
5
6
]);
int015(2, 98, [`Lower bound`
0
1E-6
200
400
600
800
1000
]);
int015(3, 102, [`Upper bound`
1E-6
200
400
600
800
1000
1001
]);
int015(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
int015(5, 105, [`Description`
``
`A`
`B`
`C`
`D`
`E`
`F`
]);
SetStatus(0);
int018;


{Atom: TimeOccupiedNorms}

sets;
AtomByName([TimeOccupiedNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TimeOccupiedNorms'. Inheriting from BaseClass.]));
CreateAtom(a, s, [TimeOccupiedNorms], 1, false);
int023([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(68);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
int015(0, 64, [0
1
2
3
4
5
6
7
]);
int015(1, 64, [`Level`
0
1
2
3
4
5
6
]);
int015(2, 98, [`Lower bound`
0
1E-6
60
120
180
240
300
]);
int015(3, 102, [`Upper bound`
1E-6
60
120
180
240
300
301
]);
int015(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
int015(5, 64, [`Description`
``
`A`
`B`
`C`
`D`
`E`
`F`
]);
SetStatus(0);
int018;
Up;


{Atom: TravelTimeNorms}

sets;
AtomByName([TravelTimeNorms], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TravelTimeNorms'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [TravelTimeNorms], 1, false);
SetAtt([AutoRescaleNorms], 17002);
SetAtt([BlendColors], 0);
SetAtt([ShowOnLayer], 1);
SetAtt([SelectRoute], 0);
int023([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(69);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(7, 5);
int015(0, 64, [0
1
2
3
4
5
6
7
]);
int015(1, 64, [`Level`
0
1
2
3
4
5
6
]);
int015(2, 98, [`Lower bound`
0
1E-6
120
240
360
480
600
]);
int015(3, 102, [`Upper bound`
1E-6
120
240
360
480
600
601
]);
int015(4, 64, [`Color`
16777215
16744448
65280
65535
33023
255
8388736
]);
int015(5, 64, [`Description`
``
`A`
`B`
`C`
`D`
`E`
`F`
]);
SetStatus(0);
int018;
Up;


{Atom: LstOutputActivityRoutes}

sets;
AtomByName([lstOutputActivityRoutes], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstOutputActivityRoutes'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [LstOutputActivityRoutes], 1, false);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(70);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: EntryToExit}

sets;
AtomByName([ActivityRoute], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ActivityRoute'. Inheriting from BaseClass.]));
CreateAtom(a, s, [EntryToExit], 1, false);
SetAtt([Interval], 30);
SetAtt([MaxNrIntervals], 40);
SetAtt([BreakDown], 0);
SetAtt([ProfileID], 0);
SetAtt([DensityInterval], 10);
SetAtt([MaxNrDensityIntervals], 20);
SetExprAtt([DensityLowerBound], [{**>= Level E**}Cell(6, DENSITYNORMS_COLUMN_LOWERBOUNDARY, atmDensityNorms)]);
SetExprAtt([DensityUpperBound], [{**No upper bound**}0]);
SetAtt([DistanceInterval], 30);
SetAtt([MaxNrDistanceIntervals], 40);
SetAtt([DelayTimeInterval], 10);
SetAtt([MaxNrDelayTimeIntervals], 40);
SetAtt([MaxContentInterval], 60);
SetTextAtt([TravelTimeSettings], []);
SetTextAtt([DistanceSettings], []);
SetTextAtt([DensityPercentageSettings], []);
SetTextAtt([DensityTimeSettings], []);
SetTextAtt([DelayTimeSettings], []);
SetTextAtt([MaxContentSettings], []);
SetAtt([SocialCostInterval], 120);
SetAtt([MaxNrSocialCostIntervals], 40);
SetTextAtt([SocialCostPerAgentSettings], []);
SetTextAtt([SocialCostTotalTimeSettings], []);
SetTextAtt([SocialCostTotalValueSettings], []);
SetAtt([AvgTravelTimePerTimeSettings], 0);
SetAtt([AvgTravelTimePerTimeInterval], 60);
int023([], 0, 4096);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(71);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetTable(2, 6);
int015(0, 64, [0
1
2
]);
int015(1, 81, [`TravelTimeType`
3331
3331
]);
int015(2, 78, [`ActivityType`
6001
6001
]);
int015(3, 72, [`ActivityGroup`
`-1`
`-1`
]);
int015(4, 64, [`LayerID`
0
0
]);
int015(5, 64, [`ActivityID`
-1
-1
]);
int015(6, 64, [`TimeIn`
0
0
]);
SetStatus(0);
int018;


{Atom: TravelTimes}

sets;
AtomByName([TravelTimes], Main);
if(not(AtomExists), Error([Cannot find mother atom 'TravelTimes'. Inheriting from BaseClass.]));
CreateAtom(a, s, [TravelTimes], 1, false);
int023([], 0, 1154);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(72);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;
Up;


{Atom: ResultPlayer}

sets;
AtomByName([ResultPlayer], Main);
if(not(AtomExists), Error([Cannot find mother atom 'ResultPlayer'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [ResultPlayer], 1, false);
SetAtt([lstUnusedAgents], 0);
SetAtt([CycleTime], 0);
SetAtt([StartTime], 0);
SetAtt([EndTime], 0);
SetAtt([Initialized], 0);
SetAtt([VisualizeSpeed], 1);
SetAtt([LastTime], 0);
SetAtt([TimeWarpTime], 0);
int042(0, 0);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(73);
SetSize(1, 1, 1);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;


{Atom: lstUnsedAgents}

sets;
AtomByName([lstUnsedAgents], Main);
if(not(AtomExists), Error([Cannot find mother atom 'lstUnsedAgents'. Inheriting from BaseClass.]));
CreateAtom(a, s, [lstUnsedAgents], 1, false);
int023([], 0, 524288);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
SetTreeIcon(pDir([Media\Icons\List.ico]));
Layer(LayerByName([DoNotDraw]));
int001(74);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;


{Atom: AgentStatistics}

sets;
AtomByName([AgentStatistics], Main);
if(not(AtomExists), Error([Cannot find mother atom 'AgentStatistics'. Inheriting from BaseClass.]));
CreateAtom(a, Up(s), [AgentStatistics], 1, false);
SetAtt([OutputStatistics], 0);
SetAtt([OverallDirectory], 0);
int023([], 0, 528384);
Set(Icon(a), 
	RegisterIcon(pDir([media\images\default.jpg]), [default.jpg]));
SetMaterial(
	RegisterMaterial([Default], 8421504, 8421504, 3289650, 0, 0.100000001490116, 0, false, false, 1, 0), 1, a);
Set(Version(a), 0);
Layer(LayerByName([DoNotDraw]));
int001(75);
LockPosition(false);
LockSize(false);
DisableIconRotation(false);
SetStatus(0);
int018;
Up;
Up;
Up;
int006(0, 3, 2, 1000000, 0, 0);

{Saved Model3D settings.}

int007;
