import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline


if __name__ == '__main__':

    plt.rcParams['font.sans-serif'] = ['SimHei']  # 如果要显示中文字体,则在此处设为：SimHei
    plt.rcParams['axes.unicode_minus'] = False  # 显示负号

    x = np.array([i for i in range(1, 51)])
    x1 = np.array([i for i in range(0, 51, 10)])
    mimic3_dropout0 = np.array([0.17052811307799678, 0.18904418560321323, 0.21339935901792634, 0.23538664891453295, 0.24595538541574305, 0.2518441127073463, 0.25563050231744294, 0.2630038233378649, 0.26425386839719717, 0.26912671772702496, 0.2742727535694422, 0.28049339320023203, 0.2835996089377363, 0.28617718537689085, 0.29255679766280795, 0.2974343304841834, 0.30135829463232183, 0.3007224969297197, 0.3061174782412035, 0.3134860251343312, 0.31124403980404536, 0.3088612060321905, 0.31665790263764737, 0.3191807823492337, 0.31714749670013925, 0.3134202523986017, 0.32092921533469526, 0.32266693963121373, 0.3208332186174422, 0.3231875345548781, 0.3204356930599083, 0.3201615820555593, 0.31994521073781873, 0.32191193917339667, 0.32409844638312413, 0.32224741300717646, 0.3210092413459559, 0.32008769625324646, 0.32291835738218505, 0.3188310152782823, 0.3160742644795578, 0.3190056199470587, 0.3179399612598851, 0.318216472830473, 0.3181775370853954, 0.3157934048330513, 0.31684602013320345, 0.3171535165258649, 0.31862142586189623, 0.316828767472627])
    mimic3_dropout02 = np.array([0.16286444921429435, 0.17678658523323823, 0.19853659449138933, 0.21838500954950468, 0.2309506288840971, 0.2393742131762955, 0.2430586140438339, 0.24702499018099183, 0.25098047046318084, 0.25032796935511636, 0.25515080766436676, 0.26034452141101905, 0.262640759003881, 0.2655480356617684, 0.26898791208246364, 0.26871059340488423, 0.2702790659214467, 0.2753038032344031, 0.2751361244856187, 0.2764219028925867, 0.2770017457387326, 0.27717827772274667, 0.27918531611949166, 0.2789982078865981, 0.2805691228357173, 0.28143517992086925, 0.28409556518980994, 0.28137294190333256, 0.2830032624229885, 0.2855199571048424, 0.28487602930167644, 0.28539965299796866, 0.2850243450369821, 0.28571746470311365, 0.2861442636635405, 0.28884905036249103, 0.28405428053340986, 0.28552488718934727, 0.2875895708797377, 0.2876673015743921, 0.28919666269237015, 0.28955575427476465, 0.2889511529058746, 0.2880402707032148, 0.2893016588594988, 0.288660473426032, 0.2900293453838321, 0.29022577629031954, 0.28900074575987444, 0.2882830123130359])
    mimic3_dropout04 = np.array([0.15874765015389572, 0.1694758039185706, 0.18470721883451408, 0.19798084547949307, 0.2126516281418948, 0.2245984890839543, 0.23196360169107946, 0.23650819616149107, 0.2395277046287324, 0.24414450393435977, 0.24791473795756447, 0.24590409747155895, 0.2513079682556513, 0.25214386597935945, 0.2526306594930059, 0.2585996301150287, 0.2620258479214493, 0.27281432336842704, 0.27957871100658177, 0.2838110606207377, 0.29454980673482417, 0.29033229998592375, 0.30153162102526726, 0.31012939699798303, 0.3072967610352185, 0.31460526347693446, 0.3180323389526119, 0.31806038708172596, 0.3193112349484766, 0.320990693830055, 0.3244067259331102, 0.3254911227094991, 0.3246766602395313, 0.32719354733651496, 0.32416093703969745, 0.3251295017374454, 0.32834616160507485, 0.3282496798912727, 0.32663062695942346, 0.3303935111434207, 0.32732317303130815, 0.3299861031453391, 0.3302908236689544, 0.33013083672596705, 0.32958416710433935, 0.3310824554026456, 0.32897127844403123, 0.33003457242271844, 0.32942131448977974, 0.3330990857814062])
    mimic3_dropout06 = np.array([0.1614410646702923, 0.16794963544502844, 0.18061939850709874, 0.19745384514972064, 0.22882825357319053, 0.27021548149043356, 0.27837396904290895, 0.29445191590313446, 0.29610264814627674, 0.3007001514332444, 0.30248283029241674, 0.3079492693126161, 0.3093485559950592, 0.31425006250280363, 0.3104738342310103, 0.31603193341086616, 0.3147363983440729, 0.310462994976369, 0.31996135247131824, 0.3202425443581869, 0.3171732632163275, 0.3198666472218204, 0.3225678033989652, 0.32334405175002023, 0.3203147002713662, 0.32269836341943425, 0.325625032478163, 0.32771018504717453, 0.32174188648784957, 0.32468844903221916, 0.3250804680159843, 0.32386863509579766, 0.32589944044505226, 0.3237469615997279, 0.32623576408238275, 0.3245927645718977, 0.3246812637576468, 0.32469855357773325, 0.32861200862942846, 0.3265213103948374, 0.32507030570215195, 0.3249237401468867, 0.3246449379665311, 0.32273771494975156, 0.3272330584604231, 0.32818269973169106, 0.32480333078522466, 0.3282774770882967, 0.3312899892585692, 0.32986681306481125])
    mimic3_dropout08 = np.array([0.16754119245811339, 0.2082568871556993, 0.2375371410175694, 0.26880700230989785, 0.2754390706560753, 0.28587454982142807, 0.28856978583976195, 0.2952678944005076, 0.2974322591913876, 0.30510362006647274, 0.3071451332133908, 0.30854600737599336, 0.3093353815388255, 0.30873009477012653, 0.3093023148098987, 0.3146502801708119, 0.31151966230257, 0.31366354384852124, 0.3163164264138859, 0.3217698804368694, 0.3190193825619273, 0.31925009303335466, 0.3192829818756304, 0.31927459261584434, 0.32013473653987407, 0.32167829718480034, 0.32312928024147003, 0.3239591296789984, 0.3222201374811941, 0.31995298352625023, 0.3214975582670582, 0.32496299530584993, 0.3235788061511141, 0.3255505504231894, 0.3206069287117745, 0.3229574189693878, 0.3242876142898493, 0.32469325196197085, 0.3271791925543631, 0.32138203979983176, 0.32395663599885766, 0.3224118508056314, 0.32458066837947785, 0.32487970714880526, 0.3248933702898985, 0.32251513591889064, 0.32516199416968117, 0.3238547785562152, 0.3233509610066068, 0.32189228367160855])

    mimic4_dropout0 = np.array([0.22603584470824695, 0.28219689116795793, 0.3227883041110027, 0.33668049592174043, 0.34811168740650056, 0.35581952264586997, 0.35609247019348284, 0.3666223115216178, 0.37014964753331847, 0.37575417169292896, 0.3727393839800179, 0.38072200510799903, 0.38254253244275344, 0.3818515026589695, 0.3811561096177138, 0.3841227228915349, 0.38988605114800096, 0.3891381454359175, 0.3840163376489387, 0.3823631764174916, 0.39429779056562564, 0.3947853831143721, 0.382501989931114, 0.3908270755729371, 0.3953651593079312, 0.387786680106689, 0.3924687899084737, 0.3920827450408044, 0.3943756885604318, 0.384837361430966, 0.3843434205339078, 0.3926073068021464, 0.390828177726254, 0.3932751321498182, 0.39057056811095986, 0.3951529766079356, 0.3897823754311641, 0.39031377806041656, 0.3877396551842792, 0.3903711409499163, 0.3893627978473892, 0.38173120597688054, 0.387614949439632, 0.38412950447376315, 0.3877612268715749, 0.3803690014620479, 0.38359997649574645, 0.38582571357171724, 0.38043282333058803, 0.38212023312702964])
    mimic4_dropout02 = np.array([0.2108352385949268, 0.24390729380340512, 0.26931998400968654, 0.30003254513646044, 0.3323669193927922, 0.3492606201977744, 0.35955194982635275, 0.36185421135055, 0.3618405373498038, 0.3709622046469107, 0.3699677929630166, 0.3756786027400406, 0.38008484621401567, 0.38329916219631943, 0.3870253763093988, 0.38260832046785437, 0.3906670144184989, 0.3876409324561205, 0.3891449241920271, 0.38848145487209973, 0.3944254023290369, 0.39142367628651675, 0.3882388668497858, 0.390784163470642, 0.3930158247204077, 0.38925717696561396, 0.39283181304858755, 0.39097901788658435, 0.39610321844624785, 0.3990690065136995, 0.39440822460721886, 0.3960710664021779, 0.3963851425968267, 0.39576326498790076, 0.3921742607303041, 0.396536486277305, 0.3973591959489026, 0.3981389057347004, 0.39646688827247034, 0.3987482558093966, 0.3975731869233379, 0.3972687794800577, 0.3960350024487844, 0.39678951580660415, 0.3913679278102529, 0.3997993371731794, 0.39738533873338266, 0.39691159272853976, 0.399135897244259, 0.39979797707079545])
    mimic4_dropout04 = np.array([0.20534208027854478, 0.22963365837355246, 0.25093446428260224, 0.2803403384318059, 0.3235454652533532, 0.34144153553618845, 0.35009904332318703, 0.3564041331253414, 0.36012946954559644, 0.3685946118788582, 0.37248386811904993, 0.37639974297603457, 0.3748561448715158, 0.38024263696637495, 0.3838002575777102, 0.38427480482331944, 0.38121734275275854, 0.389624829501178, 0.3899070040746874, 0.3867508905008631, 0.3919607375648693, 0.38700160982421966, 0.39293383182480124, 0.3909466058314722, 0.3932872539328728, 0.3887085464153331, 0.3950622546909321, 0.3948002551841215, 0.3961331098581089, 0.39237029253857386, 0.39329836180818734, 0.39614544514982153, 0.39247924549149776, 0.3932760467495815, 0.39420946862887923, 0.39712509192042045, 0.4003951668728404, 0.40129818239843895, 0.39130303342660133, 0.4002961195874534, 0.3955184759726453, 0.39653774968325595, 0.40097177789115324, 0.397404554756316, 0.39634794309119425, 0.4005996466056878, 0.398237493710398, 0.39656874400693803, 0.39858539102948415, 0.3997168695267189])
    mimic4_dropout06 = np.array([0.2025613037391252, 0.24457807525217398, 0.30648915588471665, 0.3268776596660555, 0.3394154468631753, 0.3479311436705673, 0.35207916642979054, 0.35999859169312176, 0.36518403175632386, 0.37447048957882817, 0.3738765148435892, 0.37372537473034984, 0.3800057891643359, 0.385153346266141, 0.37885534673192794, 0.3815782864848437, 0.38132210913242753, 0.3868475399593553, 0.3899909383884445, 0.38879471626866513, 0.3884629096446801, 0.3903996528077065, 0.39326292119542344, 0.38965199422377367, 0.3938779000655433, 0.39182109148438715, 0.39254328358774854, 0.39561574688021456, 0.3925469647793817, 0.3979833626067942, 0.39476113074306507, 0.3974291950327991, 0.39830956010254287, 0.3977742435044725, 0.3966855872248252, 0.39653696913777625, 0.39671643208464374, 0.3964385255655926, 0.39726616496376466, 0.39722657139463396, 0.3974704555316803, 0.39837611612403034, 0.3929087169892069, 0.39419793885811877, 0.401477334176784, 0.3956899901712652, 0.3974015606110688, 0.3964118588860752, 0.3954397834042736, 0.3999276785145655])
    mimic4_dropout08 = np.array([0.22536919790254073, 0.2870344498336406, 0.3156725566212721, 0.33238010172771415, 0.3428813295231154, 0.3521871046512096, 0.3561323219314277, 0.36752303643456624, 0.3671098518991409, 0.37293939456818176, 0.3771066701283889, 0.3801146358440326, 0.3760690788864762, 0.38419847228050663, 0.3875580246684685, 0.38676500292093696, 0.3878846705677795, 0.3891185222897085, 0.3876092082906246, 0.3932355691504361, 0.3910825898165667, 0.3932316757422663, 0.3975370925523023, 0.3962144708969293, 0.3936895067540875, 0.39690530985552086, 0.39361725950437565, 0.3900765373573028, 0.39669700618820364, 0.3947819227426463, 0.3973276804781566, 0.3927869580246489, 0.3999011759385851, 0.40053957355888065, 0.39721210716929156, 0.39826598101243854, 0.398757736130544, 0.3963745492104351, 0.39587855325814647, 0.39819473229044094, 0.396746974089397, 0.4004292086526564, 0.3969780949322487, 0.40062640347045025, 0.39811332368616625, 0.4015439564014284, 0.3996738165759679, 0.3986136103516535, 0.4006729003553398, 0.3995829899627054])

    x_smooth = np.linspace(x.min(), x.max(), 300)
    y3_1 = make_interp_spline(x, mimic3_dropout0)(x_smooth)
    y3_2 = make_interp_spline(x, mimic3_dropout02)(x_smooth)
    y3_3 = make_interp_spline(x, mimic3_dropout04)(x_smooth)
    y3_4 = make_interp_spline(x, mimic3_dropout06)(x_smooth)
    y3_5 = make_interp_spline(x, mimic3_dropout08)(x_smooth)

    y4_1 = make_interp_spline(x, mimic4_dropout0)(x_smooth)
    y4_2 = make_interp_spline(x, mimic4_dropout02)(x_smooth)
    y4_3 = make_interp_spline(x, mimic4_dropout04)(x_smooth)
    y4_4 = make_interp_spline(x, mimic4_dropout06)(x_smooth)
    y4_5 = make_interp_spline(x, mimic4_dropout08)(x_smooth)

    # label在图示(legend)中显示。若为数学公式,则最好在字符串前后添加"$"符号
    # color：b:blue、g:green、r:red、c:cyan、m:magenta、y:yellow、k:black、w:white、、、
    # 线型：-  --   -.  :    ,
    # marker：.  ,   o   v    <    *    +    1

    plt.figure(figsize=(10, 5))
    # plt.grid(linestyle="--")  # 设置背景网格线为虚线
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    # plt.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.1, wspace=0)
    plt.subplot(1, 2, 1)
    # plt.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.1)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.plot(x_smooth, y3_1, color="darkorange", label="dropout=0", linewidth=1.5)
    plt.plot(x_smooth, y3_2, color="forestgreen", label="dropout=0.2", linewidth=1.5)
    plt.plot(x_smooth, y3_3, color="tab:blue", label="dropout=0.4", linewidth=1.5)
    plt.plot(x_smooth, y3_4, color="firebrick", label="dropout=0.6", linewidth=1.5)
    plt.plot(x_smooth, y3_5, color="tab:grey", label="dropout=0.8", linewidth=1.5)


    group_labels = ['0', '10', '20', '30', '40', ' 50']  # x轴刻度的标识
    plt.xticks(x1, group_labels, fontsize=12)  # 默认字体大小为10
    plt.yticks(fontsize=12)
    # plt.title("(a)MIMIC-III数据集", fontsize=8, y=-0.2)  # 默认字体大小为12
    plt.xlabel("迭代次数", fontsize=12)
    plt.ylabel("Recall@5", fontsize=12)
    plt.xlim(0, 51)  # 设置x轴的范围
    plt.ylim(0.15, 0.35)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12)  # 设置图例字体的大小和粗细

    # 图2
    plt.subplot(1, 2, 2)
    plt.grid(linestyle="--")  # 设置背景网格线为虚线
    # plt.subplots_adjust(left=0.125, right=0.9, top=0.9, bottom=0.1)
    # ax = plt.gca()
    # ax.spines['top'].set_visible(False)  # 去掉上边框
    # ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.plot(x_smooth, y4_1, color="darkorange", label="dropout=0", linewidth=1.5)
    plt.plot(x_smooth, y4_2, color="forestgreen", label="dropout=0.2", linewidth=1.5)
    plt.plot(x_smooth, y4_3, color="tab:blue", label="dropout=0.4", linewidth=1.5)
    plt.plot(x_smooth, y4_4, color="firebrick", label="dropout=0.6", linewidth=1.5)
    plt.plot(x_smooth, y4_5, color="tab:grey", label="dropout=0.8", linewidth=1.5)

    group_labels = ['0', '10', '20', '30', '40', '50']  # x轴刻度的标识
    plt.xticks(x1, group_labels, fontsize=12)  # 默认字体大小为10
    plt.yticks(fontsize=12)
    # plt.title("(b)MIMIC-IV数据集", fontsize=8, y=-0.2)  # 默认字体大小为12
    plt.xlabel("Iterations", fontsize=10)
    plt.ylabel("Recall@5", fontsize=10)
    plt.xlim(0, 51)  # 设置x轴的范围
    plt.ylim(0.3, 0.42)

    # plt.legend()          #显示各曲线的图例
    plt.legend(loc=0, numpoints=1)
    leg = plt.gca().get_legend()
    ltext = leg.get_texts()
    plt.setp(ltext, fontsize=12)  # 设置图例字体的大小和粗细

    plt.savefig('./mimic-dropout.svg', format='svg')  # 建议保存为svg格式,再用在线转换工具转为矢量图emf后插入word中
    plt.show()
