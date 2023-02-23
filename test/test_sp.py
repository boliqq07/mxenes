import unittest

from mxene.core.mxenes import MXene

p1 = """SC111 file generated by VASPKIT         
   1.00000000000000     
     8.6444418542189698    0.0000000000000000    0.0000000000000000
    -4.3222209271094849    7.4863131239464709    0.0000000000000000
     0.0000000000000000    0.0000000000000000   25.0000000000000000
   C    O    Mo   Ag
     9    18    17     1
Direct
  0.9988482436274854  0.0011517696432331  0.4981942333686641
 -0.0021584830521210  0.3322540524119901  0.4994233623996874
  0.9988481913228758  0.6643632370770273  0.4981942743290561
  0.3356367580660916  0.0011518113666584  0.4981942654570685
  0.3274917916249386  0.3216501243379541  0.4984925848656721
  0.3274915915981537  0.6725083624736091  0.4984925402067758
  0.6677459715534603  0.0021584963398390  0.4994233290427501
  0.6677461034490531  0.3322539307423895  0.4994234351716516
  0.6783498657551602  0.6725081802794719  0.4984925711458592
  0.0003624307066631 -0.0003624048330602  0.6015217166895078
  0.0024437209335566  0.3345554712360767  0.6020837225351339
  0.0003627724600637  0.6673921297640252  0.6015217601085804
  0.3326079579059089 -0.0003627196224361  0.6015217508882882
  0.3089528365753564  0.2845721107628503  0.6060556343236749
  0.3089526011649019  0.6910475313598480  0.6060556244706301
  0.6654445795041360 -0.0024437262718910  0.6020836901606103
  0.6654453884830810  0.3345547069891532  0.6020837734992879
  0.7154279195202583  0.6910472524957414  0.6060556196238929
  0.9981341837620973  0.0018658290131302  0.3950334085336317
  0.0018990324739961  0.3342829882636837  0.3963696339045492
  0.9981337154679890  0.6629346374580957  0.3950334195364090
  0.3370654246744899  0.0018663347987411  0.3950334385198171
  0.3361015298008420  0.3388686212812104  0.3979494669651227
  0.3361003040681638  0.6638995453910508  0.3979494429949195
  0.6657170098203345 -0.0018991139236723  0.3963696007209760
  0.6657174976469993  0.3342826035840431  0.3963697036969043
  0.6611312069781405  0.6638983161419140  0.3979494509102676
  0.1109405034141000  0.2181170248719102  0.5532867684675106
  0.1109405044381889  0.5594902519551500  0.5532867851822625
  0.1111110345314159  0.8888889402074721  0.5522432107802296
  0.4405099126106943  0.2181169031258887  0.5532868230437329
  0.4405097065442543  0.8890594607477907  0.5532867499866047
  0.7777778438411266  0.2222220637827348  0.5542185309412858
  0.7818830021775869  0.5594900599092898  0.5532868346981644
  0.7818828899390217  0.8890593944850151  0.5532867434916685
  0.2215796147179871  0.1098258433686106  0.4434362470897877
  0.2222894509822158  0.4444781647804820  0.4474458945161550
  0.2215794019235662  0.7784206731576744  0.4434362502719968
  0.5573818652498671  0.1120242056778165  0.4440748205758872
  0.5555221322616354  0.4444779266060180  0.4474459772744625
  0.5555218604188507  0.7777105465050893  0.4474458801349105
  0.8879758107542223  0.1120242350164650  0.4440747489951094
  0.8879758943741985  0.4426182791830430  0.4440748403982482
  0.8901742190501134  0.7784204533847097  0.4434362388699503
  0.4444442938788752  0.5555555816741591  0.5835757412126047
  """

p2 = """SC111 file generated by VASPKIT         
   1.00000000000000     
     8.6444418542189698    0.0000000000000000    0.0000000000000000
    -4.3222209271094849    7.4863131239464709    0.0000000000000000
     0.0000000000000000    0.0000000000000000   25.0000000000000000
   C    O    Mo   Ag  H
     9    18    17     1  1
Direct
  0.9988482436274854  0.0011517696432331  0.4981942333686641
 -0.0021584830521210  0.3322540524119901  0.4994233623996874
  0.9988481913228758  0.6643632370770273  0.4981942743290561
  0.3356367580660916  0.0011518113666584  0.4981942654570685
  0.3274917916249386  0.3216501243379541  0.4984925848656721
  0.3274915915981537  0.6725083624736091  0.4984925402067758
  0.6677459715534603  0.0021584963398390  0.4994233290427501
  0.6677461034490531  0.3322539307423895  0.4994234351716516
  0.6783498657551602  0.6725081802794719  0.4984925711458592
  0.0003624307066631 -0.0003624048330602  0.6015217166895078
  0.0024437209335566  0.3345554712360767  0.6020837225351339
  0.0003627724600637  0.6673921297640252  0.6015217601085804
  0.3326079579059089 -0.0003627196224361  0.6015217508882882
  0.3089528365753564  0.2845721107628503  0.6060556343236749
  0.3089526011649019  0.6910475313598480  0.6060556244706301
  0.6654445795041360 -0.0024437262718910  0.6020836901606103
  0.6654453884830810  0.3345547069891532  0.6020837734992879
  0.7154279195202583  0.6910472524957414  0.6060556196238929
  0.9981341837620973  0.0018658290131302  0.3950334085336317
  0.0018990324739961  0.3342829882636837  0.3963696339045492
  0.9981337154679890  0.6629346374580957  0.3950334195364090
  0.3370654246744899  0.0018663347987411  0.3950334385198171
  0.3361015298008420  0.3388686212812104  0.3979494669651227
  0.3361003040681638  0.6638995453910508  0.3979494429949195
  0.6657170098203345 -0.0018991139236723  0.3963696007209760
  0.6657174976469993  0.3342826035840431  0.3963697036969043
  0.6611312069781405  0.6638983161419140  0.3979494509102676
  0.1109405034141000  0.2181170248719102  0.5532867684675106
  0.1109405044381889  0.5594902519551500  0.5532867851822625
  0.1111110345314159  0.8888889402074721  0.5522432107802296
  0.4405099126106943  0.2181169031258887  0.5532868230437329
  0.4405097065442543  0.8890594607477907  0.5532867499866047
  0.7777778438411266  0.2222220637827348  0.5542185309412858
  0.7818830021775869  0.5594900599092898  0.5532868346981644
  0.7818828899390217  0.8890593944850151  0.5532867434916685
  0.2215796147179871  0.1098258433686106  0.4434362470897877
  0.2222894509822158  0.4444781647804820  0.4474458945161550
  0.2215794019235662  0.7784206731576744  0.4434362502719968
  0.5573818652498671  0.1120242056778165  0.4440748205758872
  0.5555221322616354  0.4444779266060180  0.4474459772744625
  0.5555218604188507  0.7777105465050893  0.4474458801349105
  0.8879758107542223  0.1120242350164650  0.4440747489951094
  0.8879758943741985  0.4426182791830430  0.4440748403982482
  0.8901742190501134  0.7784204533847097  0.4434362388699503
  0.4444442938788752  0.5555555816741591  0.5535757412126047
  0.3089528365753564  0.2845721107628503  0.6360556343236749
  """

p1 = MXene.from_str(p1, fmt="poscar")
p2 = MXene.from_str(p2, fmt="poscar")

class MyTestCase(unittest.TestCase):

    def test_force_plane(self):
        # label1 = p1.split_layer(force_plane=True, tol=0.5)
        label2 = p2.split_layer(force_plane=True, tol=0.5)
        label2 = p2.split_layer(force_plane=True,reverse=False)

    def test_force_finite(self):
        # label1 = p1.split_layer(force_plane=True, tol=0.5)
        label2 = p2.split_layer(force_plane=True, tol=0.5)
        label2 = p2.split_layer(force_plane=True,reverse=False)

if __name__ == '__main__':
    unittest.main()
