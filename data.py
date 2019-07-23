import numpy as np
import matplotlib.pyplot as plt

'''
# AFH s = 1/2 chain

D_max = np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
t_list = [0.1, 0.01, 0.001, 0.0001, 0.00001]
iterations = 1000
d = 2
p = 2
J = -1
imat = np.array([[1, 1],
                 [1, 1]])
smat = np.array([[2, 1],
                 [2, 1]])
#Hamiltonian = -J * np.real(np.kron(sx, sx) + np.kron(sy, sy) + np.kron(sz, sz))
gs_energy_per_site = [-0.4279027320644977, -0.4357778474711217, -0.44105237214050586, -0.4420421043243941, -0.44248914059184913, -0.4426419886222407, -0.4427701423171405, -0.44288225027867717, -0.44299788414494534, -0.44308764142433765, -0.44311382973770275, -0.443127051421859, -0.44313210229879896, -0.44313527791307084, -0.4431369088821772, -0.44313799019369265, -0.4431387714497798, -0.4431392604845007, -0.44313960806966257]

D_max_and_convergence_time = np.array([[2., 3., 4., 5., 6., 7., 8., 9., 10., 15., 20., 25., 30., 35., 40., 45., 50., 55., 60.], 
                    [331., 439., 1083., 1204., 1237., 1353., 1261., 827., 1285., 1153., 1733., 939., 1059., 1140.,
                     1836., 1236., 1230., 1300., 1326.]])
plt.figure()
plt.title('Antiferromagnetic Heisenberg S=1/2 chain\n SU iterations=2000 at each dt=[1e-1, 1e-2, 1e-3, 1e-4, 1e-5]')
plt.plot(D_max_and_convergence_time[0, :] ** (-1), gs_energy_per_site, 'o', ls='-')
plt.xlabel('1/D')
plt.ylabel('e')
plt.grid()
plt.show()                     
'''

## ITF (ZZX) 16 spins PEPS with bond dimension Dmax = 2

h = [0.        , 0.05050505, 0.1010101 , 0.15151515, 0.2020202 ,
       0.25252525, 0.3030303 , 0.35353535, 0.4040404 , 0.45454545,
       0.50505051, 0.55555556, 0.60606061, 0.65656566, 0.70707071,
       0.75757576, 0.80808081, 0.85858586, 0.90909091, 0.95959596,
       1.01010101, 1.06060606, 1.11111111, 1.16161616, 1.21212121,
       1.26262626, 1.31313131, 1.36363636, 1.41414141, 1.46464646,
       1.51515152, 1.56565657, 1.61616162, 1.66666667, 1.71717172,
       1.76767677, 1.81818182, 1.86868687, 1.91919192, 1.96969697,
       2.02020202, 2.07070707, 2.12121212, 2.17171717, 2.22222222,
       2.27272727, 2.32323232, 2.37373737, 2.42424242, 2.47474747,
       2.52525253, 2.57575758, 2.62626263, 2.67676768, 2.72727273,
       2.77777778, 2.82828283, 2.87878788, 2.92929293, 2.97979798,
       3.03030303, 3.08080808, 3.13131313, 3.18181818, 3.23232323,
       3.28282828, 3.33333333, 3.38383838, 3.43434343, 3.48484848,
       3.53535354, 3.58585859, 3.63636364, 3.68686869, 3.73737374,
       3.78787879, 3.83838384, 3.88888889, 3.93939394, 3.98989899,
       4.04040404, 4.09090909, 4.14141414, 4.19191919, 4.24242424,
       4.29292929, 4.34343434, 4.39393939, 4.44444444, 4.49494949,
       4.54545455, 4.5959596 , 4.64646465, 4.6969697 , 4.74747475,
       4.7979798 , 4.84848485, 4.8989899 , 4.94949495, 5.        ]

E = [-1.999999998019237,
 -2.0003188437951436,
 -2.0012753914357337,
 -2.002869674195031,
 -2.005101739398388,
 -2.007971661471799,
 -2.0114795343686147,
 -2.015625478832349,
 -2.0204096210431897,
 -2.0258321237203285,
 -2.031893165208105,
 -2.0385929560925997,
 -2.0459317097119545,
 -2.0539096787639703,
 -2.062527136275599,
 -2.071784377852479,
 -2.0816817262129867,
 -2.0922195360516156,
 -2.103398167407654,
 -2.115218028671447,
 -2.1276795501592374,
 -2.1407831917760514,
 -2.154529451608693,
 -2.168918834101308,
 -2.18395191888312,
 -2.199629281252321,
 -2.2159515542437975,
 -2.2329194163036297,
 -2.2505335613863187,
 -2.2687947480444715,
 -2.287703776458074,
 -2.3072614975017243,
 -2.327468803803361,
 -2.348326648120536,
 -2.369836046095782,
 -2.3919980840718944,
 -2.414813886932319,
 -2.4382846757821235,
 -2.4624117549978584,
 -2.487196482209648,
 -2.512640338906403,
 -2.5387448701679984,
 -2.5655117560018224,
 -2.592942775005244,
 -2.621039831694948,
 -2.649804963098192,
 -2.6792403605377415,
 -2.7093483508412635,
 -2.7401314664542538,
 -2.771592402545372,
 -2.803734094754696,
 -2.836559684677134,
 -2.87007257306636,
 -2.9042764527054135,
 -2.939175311619847,
 -2.974773516195776,
 -3.0110757794871743,
 -3.048087288821064,
 -3.085813675377505,
 -3.1242611786115195,
 -3.163436598134311,
 -3.2033474943339475,
 -3.244002160469299,
 -3.3458609446771352,
 -3.329386497032038,
 -3.376855809339266,
 -3.424429178868999,
 -3.472100470478813,
 -3.5198643786547823,
 -3.567715994981635,
 -3.615650803043181,
 -3.6636645303697577,
 -3.711753524268589,
 -3.7599139755408344,
 -3.8081425794725607,
 -3.856436140932207,
 -3.9047917695167027,
 -3.953206695441173,
 -4.0016783665499505,
 -4.050204319079307,
 -4.0987823060994835,
 -4.147410184508208,
 -4.1960859082045525,
 -4.244807608886592,
 -4.293573479081207,
 -4.342381819659656,
 -4.391231027883621,
 -4.440119584443681,
 -4.489046041745685,
 -4.5380090420109465,
 -4.587007289371839,
 -4.636039545456518,
 -4.685104641813141,
 -4.734201476526963,
 -4.783328973958717,
 -4.832486121276467,
 -4.881671964017617,
 -4.93088558180012,
 -4.9801260938151914,
 -5.029392668271674]

mx = [3.040004207042178e-05,
 0.012657775705678581,
 0.025283902727358233,
 0.03789633887070077,
 0.05053586810918752,
 0.06318693280640403,
 0.07584662325016858,
 0.08851645855937079,
 0.1012076801053397,
 0.11391702437750181,
 0.12664746491704867,
 0.13939959668054536,
 0.1521785829502324,
 0.16498575845467345,
 0.17782323247918053,
 0.19069369597134567,
 0.2035994232469958,
 0.21654253973440046,
 0.22952661565903365,
 0.24255373097881633,
 0.2556264630158518,
 0.2687474900598132,
 0.2819192663100118,
 0.29514517594095263,
 0.30842728374757,
 0.3217689261310835,
 0.3351728761787923,
 0.3486418861637134,
 0.36217928463874804,
 0.37578802215530827,
 0.38947122397847067,
 0.4032320744528188,
 0.4170739665389328,
 0.43100029638966364,
 0.445014496991938,
 0.45912002331798996,
 0.47332072506796996,
 0.48762029376714516,
 0.5020224990807984,
 0.5165314795388147,
 0.5311512104859568,
 0.5458860530850117,
 0.5607402765957447,
 0.5757184033662219,
 0.5908250846323246,
 0.6060651353800885,
 0.6214434869460522,
 0.6369653701284647,
 0.6526360230566379,
 0.6684610145142188,
 0.6844459540157058,
 0.7005967692236243,
 0.7169195651926836,
 0.7334206080515684,
 0.7501064420763375,
 0.766983692489899,
 0.7840593456032017,
 0.8013404195876741,
 0.818834263551059,
 0.8365481543670511,
 0.8544896891881641,
 0.8726663172137328,
 0.8910856986255724,
 0.9104878935002068,
 0.9226050261476502,
 0.9247796589313504,
 0.9268452734471844,
 0.9288108695863777,
 0.9306833862065634,
 0.9324691868137547,
 0.9341718999560376,
 0.9358025128544953,
 0.9373634086199782,
 0.9388560565765397,
 0.9402864212322716,
 0.9416580127205412,
 0.942974506914173,
 0.9442389715597501,
 0.9454544836635157,
 0.9466235608977647,
 0.9477488290305791,
 0.9488326156497392,
 0.9498770454830896,
 0.9508842201011605,
 0.9518560032894712,
 0.9527941638287147,
 0.9537003610550907,
 0.9545761444888281,
 0.9554229569750329,
 0.9562421722546908,
 0.9570350698135779,
 0.9578028477015309,
 0.9585466388228334,
 0.9592675234562666,
 0.959966490696611,
 0.9606444869367781,
 0.9613024173675686,
 0.9619411272729074,
 0.9625614136271552,
 0.9631640397322649]

mz = [0.9999999994881951,
 0.9999198845018418,
 -0.9996802700061371,
 0.9992814707687251,
 -0.9987215984299356,
 0.9980001237906565,
 -0.9971162027395464,
 -0.9960686021556179,
 -0.9948548610641994,
 -0.9934734559429087,
 -0.9919220668988578,
 0.9901983896559561,
 0.9882991922574946,
 0.9862214309796826,
 0.9839616481591882,
 -0.9815159433300111,
 -0.9788801308368805,
 0.9760496928284162,
 -0.9730193740451502,
 0.9697838243037551,
 0.9663371333563597,
 -0.9626728821011016,
 0.958784207531934,
 -0.9546634000066516,
 0.9503025806696924,
 0.9456928102798186,
 -0.9408246451526623,
 0.9356879133242034,
 0.9302713946890975,
 -0.9245631313054105,
 0.9185501235645,
 -0.9122182814388873,
 -0.9055522372388534,
 -0.8985353197179439,
 -0.8911494015482655,
 0.8833747516195032,
 0.8751896136497438,
 0.8665703181960862,
 0.8574908976044925,
 -0.847922572646066,
 -0.8378337725982539,
 -0.827189311416787,
 0.8159502493192494,
 -0.8040730173349688,
 0.7915087765111561,
 -0.7782024843464846,
 -0.7640918260009716,
 0.7491056227156869,
 0.7331623683263301,
 0.7161675986027665,
 -0.6980112561371146,
 0.6785634065818932,
 -0.6576690515620207,
 -0.6351407405337246,
 0.6107478080565872,
 -0.5842011181784152,
 -0.5551287427570915,
 0.5230383172605524,
 -0.4872513602965126,
 0.4467845577075968,
 0.40010499988974385,
 0.34455419341885524,
 0.2746094301426537,
 0.29303849665956416,
 -0.0025445048715424723,
 0.001380758869715973,
 0.001149528417770748,
 0.0010365440827783902,
 0.0011378646580505673,
 0.001378825403323253,
 -0.0029666782427992698,
 0.002485846358620066,
 0.0007762611250847363,
 0.0005742998009567208,
 0.000128503262417662,
 -0.00040184396147662367,
 0.0003998436310032751,
 0.0005637630502260168,
 0.00023235990716969496,
 -0.0002813107835890206,
 0.00021125477454197658,
 5.4388647037817146e-05,
 0.0002362007376857559,
 0.00016593286583400913,
 -0.00013659504781755893,
 0.00013406547656605284,
 0.00011229590165785378,
 7.672147179004971e-05,
 0.0001212212506029276,
 -0.00013389655017397104,
 0.0001693183906359767,
 0.00013386401059842072,
 -0.00018469880693039844,
 -6.548368914112646e-05,
 -1.796233709780776e-05,
 5.8378018653360245e-05,
 8.973393356251603e-05,
 -5.4765622622789705e-05,
 -7.6173906518816e-05,
 1.317912449167796e-05]

mx_exact = [nan,
 0.012656933661578848,
 0.025282783821168144,
 0.03789597052410022,
 0.050536163516105403,
 0.06318613667725467,
 0.07584688714973772,
 0.08851651753325007,
 0.10120779593383196,
 0.11391734546285859,
 0.1266481889146852,
 0.13940062724718572,
 0.1521802754394648,
 0.16498824901894502,
 0.17782692337570016,
 0.19069896247176388,
 0.2036068481423413,
 0.21655280631578258,
 0.2295405137902865,
 0.24257223390304378,
 0.25565079789456746,
 0.26877911456876014,
 0.2819599467695859,
 0.2951969578857932,
 0.30849264631530937,
 0.32185075371169597,
 0.3352745547791988,
 0.34876737947083203,
 0.362333178443603,
 0.3759756441597685,
 0.3896987302321776,
 0.40350655232322646,
 0.41740354940853586,
 0.4313942936530156,
 0.4454835453586449,
 0.45967625174839916,
 0.4739778946385922,
 0.4883940108860799,
 0.502930415532636,
 0.5175934752133775,
 0.5323896394870058,
 0.5473259290260322,
 0.5624095031549217,
 0.5776479369892027,
 0.5930490597434523,
 0.6086208977272407,
 0.6243714925825703,
 0.6403088495292603,
 0.6564403658952132,
 0.6727726651946775,
 0.6893107076814848,
 0.7060571653297913,
 0.7230111282078043,
 0.7401665555558397,
 0.7575104185505735,
 0.7750200962303467,
 0.7926607213395813,
 0.810381586310349,
 0.8281128694672056,
 0.8457618445062902,
 0.863210350544686,
 0.8803127904693564,
 0.8968968531183159,
 0.8984899959367367,
 0.9239259485319196,
 0.9278540131183252,
 0.9314486973084334,
 0.9347489095494348,
 0.9377874490609031,
 0.9405927093073181,
 0.9431879416116412,
 0.9455982337909211,
 0.9478406187236306,
 0.9499297162384097,
 0.9518810982012447,
 0.9537072740604444,
 0.9554196369606462,
 0.9570280721963627,
 0.9585414469569332,
 0.9599675839218533,
 0.9613135796667238,
 0.962585754394564,
 0.9637897473690891,
 0.964930730501211,
 0.9660133228920286,
 0.9670417124796267,
 0.9680197072554221,
 0.9689507698321167,
 0.9698380473651346,
 0.9706844301903228,
 0.9714925578960418,
 0.9722648336509205,
 0.9730034899641313,
 0.9737105698655143,
 0.9743879511593287,
 0.9750373650821199,
 0.9756604270501112,
 0.9762586200163024,
 0.9768333222710499,
 0.9773858164485614]

mz_exact = [nan,
 0.9999198962140402,
 -0.9996803148240825,
 0.9992815671540793,
 -0.9987218440518393,
 0.9980008116160493,
 -0.997117506459152,
 -0.9960710534226972,
 -0.9948590525244572,
 -0.9934801742644174,
 -0.9919323088627859,
 0.9902134355435793,
 0.988320543095129,
 0.9862509263169608,
 0.9840014243835649,
 -0.9815685143469968,
 -0.9789483771723797,
 0.9761369126126169,
 -0.9731293250709551,
 0.9699207359638593,
 0.9665057248096414,
 -0.9628784030107381,
 0.95903244800948,
 -0.9549607426946617,
 0.9506559709699193,
 0.9461098074609616,
 -0.9413134094088638,
 0.9362571966864914,
 0.9309305478369643,
 -0.9253220546948429,
 0.9194192264609398,
 -0.9132084154786294,
 -0.9066746070870224,
 -0.8998013543303348,
 -0.8925705793527408,
 0.8849623718391303,
 0.876954539042542,
 0.8685225861723584,
 0.8596392529450223,
 -0.850273909665882,
 -0.840392367892084,
 -0.8299559359997166,
 0.8189209937958308,
 -0.8072378869358678,
 0.7948499404116356,
 -0.7816921392403375,
 -0.7676895997586883,
 0.752755498216989,
 0.7367890142714474,
 0.719672206976766,
 -0.7012668658791109,
 0.6814100065206674,
 -0.6599088048109477,
 -0.6365340819131567,
 0.6110116493202742,
 -0.5830110748628123,
 -0.5521284796902444,
 0.5178606810309014,
 -0.4795587924073249,
 0.43634111007077486,
 0.3869027836272381,
 0.3290448480492868,
 0.258191522355904,
 0.2621600808050427,
 -0.002303578091312532,
 0.0012361354797814311,
 0.0010184931730704005,
 0.0009095470838017631,
 0.0009894790025095081,
 0.0011889306181030947,
 -0.002538639288483924,
 0.0021099537262936332,
 0.0006545438572205054,
 0.00048130505657578243,
 0.00010605984873610141,
 -0.0003323870391898163,
 0.0003290064175472251,
 0.00047253225912266073,
 0.00018951444128473483,
 -0.00022794771348846836,
 0.00017041055863681526,
 4.920060615261055e-05,
 0.000184861625531837,
 0.00013205254269548492,
 -0.00010814725229010795,
 0.00010533020811282157,
 8.837763151180902e-05,
 5.963059408703268e-05,
 9.474955348974851e-05,
 -0.00010439147553888309,
 0.00012582152104017124,
 0.00010215331922175475,
 -0.00014379417925626195,
 -5.033111128396821e-05,
 -6.411004779391372e-06,
 4.473273861083797e-05,
 6.49537675171592e-05,
 -4.1513533780161355e-05,
 -5.7629905962078537e-05,
 9.780661174912598e-06]

mx_graph = [3.0400034034012337e-05,
 0.012657774083762154,
 0.025283897352617837,
 0.03789633771279055,
 0.05053587103329228,
 0.06318694863018827,
 0.07584667277175094,
 0.08851658898669404,
 0.10120793666543729,
 0.113917491616749,
 0.12664826028617515,
 0.1394008946822729,
 0.15218059721734775,
 0.16498877692032754,
 0.177827626019067,
 0.1906999304343013,
 0.20360807907903894,
 0.21655433581370437,
 0.22954240880901186,
 0.2425745591861661,
 0.25565356414260787,
 0.26878232829674054,
 0.28196356889952157,
 0.29520094513490946,
 0.3084968665037171,
 0.32185502366629726,
 0.33527860437124957,
 0.34877082865168074,
 0.3623355291861159,
 0.37597623572352623,
 0.38969671614889384,
 0.4035008722592778,
 0.4173928904481722,
 0.4313770550656256,
 0.44545779226661886,
 0.4596396697411227,
 0.47392776087425975,
 0.4883271411309019,
 0.5028431315616475,
 0.517481590294943,
 0.5322484448808841,
 0.547150231845096,
 0.5621936927239417,
 0.5773861360586547,
 0.5927353759665019,
 0.6082498282223522,
 0.6239385461747691,
 0.6398114826081249,
 0.6558793479321758,
 0.6721540530764263,
 0.6886486444990307,
 0.7053778032801346,
 0.7223580489192429,
 0.7396081638630667,
 0.7571498928018363,
 0.7750085886054201,
 0.7932146255179424,
 0.8118047861671807,
 0.8308250451972614,
 0.8503340685567635,
 0.870409801344989,
 0.8911595025921202,
 0.9127388981183963,
 0.9106847174267312,
 0.9512230216357602,
 0.9529932821883249,
 0.9546655601048541,
 0.9562488144020403,
 0.9577495512089969,
 0.9591737269455527,
 0.9605245559465446,
 0.9618127044936973,
 0.9630400401785824,
 0.9642077883294982,
 0.9653215469584957,
 0.9663845372219964,
 0.9674001033473562,
 0.9683710306155905,
 0.9693001215525271,
 0.9701896383243516,
 0.97104194903656,
 0.971859143100031,
 0.9726431320552098,
 0.9733957874025678,
 0.9741187688047152,
 0.9748136494330777,
 0.9754818993458441,
 0.9761248874538926,
 0.9767438872891696,
 0.9773401046499166,
 0.9779146600418156,
 0.978468606691773,
 0.9790029272907037,
 0.9795185642928907,
 0.9800163748017036,
 0.9804971813228684,
 0.9809617623166589,
 0.9814108479715739,
 0.9818451225444594,
 0.982265240074967]

mz_graph = [0.9999999994882267,
 0.9999198857418281,
 -0.9996802891775227,
 0.9992815659281545,
 -0.9987218990565364,
 0.9980008583012465,
 -0.9971177267905705,
 -0.9960714255753074,
 -0.9948596856587442,
 -0.9934811980297129,
 -0.9919338919433622,
 0.9902157389680164,
 0.9883238263980044,
 0.9862554555047442,
 0.9840075500045482,
 -0.9815766288750183,
 -0.9789589621309901,
 0.9761505236431443,
 -0.973146603811452,
 0.9699424289466805,
 0.9665327138974908,
 -0.9629117126721958,
 0.9590732800234315,
 -0.9550104935074715,
 0.9507162896894946,
 0.9461826286791555,
 -0.941401014375262,
 0.9363622828595144,
 0.9310563083345847,
 -0.9254722853436431,
 0.9195984465937153,
 -0.913422014494846,
 -0.9069290270703582,
 -0.9001043066231397,
 -0.8929313121788385,
 0.885391993371527,
 0.877466404784776,
 0.869132779427924,
 0.8603671622713226,
 -0.851142933092945,
 -0.8414307718546212,
 -0.8311978975748853,
 0.8204078682089758,
 -0.8090197419710499,
 0.7969874076238814,
 -0.7842586341314599,
 -0.7707739567734416,
 0.7564650693049886,
 0.7412532155573972,
 0.7250465053431111,
 -0.7077370202497382,
 0.6891963161304749,
 -0.6692697663980898,
 -0.6477684927228571,
 0.6244575850217458,
 -0.5990390718340419,
 -0.571124893717762,
 0.5401943015641733,
 -0.5055196444024574,
 0.4660309486531526,
 0.42003709783105403,
 0.364569346354749,
 0.29341215980085167,
 0.2932113139389052,
 -0.0027475139779644005,
 0.0014472082729980358,
 0.0011740423462810769,
 0.0010349016346287467,
 0.0011135837417591103,
 0.001325735069693676,
 -0.0028078854282859826,
 0.0023198366323981465,
 0.000715364886573127,
 0.0005232531996209738,
 0.00011613689613465539,
 -0.00035904905443056787,
 0.0003542281431866551,
 0.0004947630177467993,
 0.00020280251303202157,
 -0.0002439630019117657,
 0.00018211423108401362,
 4.597397837272435e-05,
 0.00020212586039655353,
 0.00014087158135925662,
 -0.00011550356400722578,
 0.0001129670683929912,
 9.416859292011096e-05,
 6.414608548099138e-05,
 0.00010096029706172654,
 -0.00011115462848802424,
 0.00014079885579327406,
 0.000110607931633349,
 -0.0001516171806087395,
 -5.378865994396659e-05,
 -1.6306653208147948e-05,
 4.772285019380082e-05,
 7.387213032549605e-05,
 -4.4631366676338435e-05,
 -6.191621386372054e-05,
 1.0754885317931784e-05]

time_to_converge = [  14.,   14.,   14.,   16.,   16.,   16.,   16.,   18.,   18.,
         18.,   18.,   20.,   20.,   20.,   20.,   20.,   20.,   22.,
         22.,   22.,   22.,   22.,   24.,   22.,   24.,   24.,   24.,
         26.,   26.,   26.,   26.,   26.,   28.,   28.,   28.,   30.,
         30.,   30.,   32.,   32.,   34.,   34.,   36.,   36.,   36.,
         40.,   40.,   46.,   44.,   44.,   48.,   52.,   56.,   54.,
         70.,   68.,   70.,   78.,  106.,   96.,  124.,  152.,  304.,
       1140.,  384.,  112.,   86.,   40.,   46.,   34.,   20.,   16.,
         36.,   22.,   16.,   20.,   26.,   14.,   20.,   22.,   22.,
         14.,   14.,   20.,   18.,   16.,   20.,   14.,   18.,   16.,
         12.,   14.,   12.,   16.,   12.,   16.,   12.,   14.,   14.,
         12.]

D_max = 2

plt.figure()
plt.title('2D Quantum Ising Model in a transverse field at Dmax = ' + str(D_max))
plt.subplot()
color = 'tab:red'
plt.xlabel('h')
plt.ylabel('Energy per site', color=color)
plt.plot(h, E, color=color)
plt.grid()
plt.tick_params(axis='y', labelcolor=color)
plt.twinx()  # instantiate a second axes that shares the same x-axis
color = 'tab:blue'
plt.ylabel('# of iterations for energy convergence', color=color)  # we already handled the x-label with ax1
plt.plot(h, time_to_converge, color=color)
plt.tick_params(axis='y', labelcolor=color)
plt.tight_layout()  # otherwise the right y-label is slightly clipped
plt.grid()
plt.show()

plt.figure()
plt.plot(h, mx, 'go', markersize=3)
plt.plot(h, np.abs(np.array(mz)), 'bo', markersize=3)
plt.plot(h, mx_exact, 'r-', linewidth=2)
plt.plot(h, np.abs(np.array(mz_exact)), 'y-', linewidth=2)
plt.plot(h, mx_graph, 'cv', markersize=5)
plt.plot(h, np.abs(np.array(mz_graph)), 'mv', markersize=5)
plt.title('magnetization vs h at Dmax = ' + str(D_max))
plt.xlabel('h')
plt.ylabel('Magnetization')
plt.legend(['mx', '|mz|', 'mx exact', '|mz| exact', 'mx DEnFG', '|mz| DEnFG'])
plt.grid()
plt.show()