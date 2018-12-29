from dragonfly import Grammar, CompoundRule, Integer, Choice, Repetition, Optional

heroes = ["Abathur","Alarak","Alexstrasza","Ana","Anubarak","Artanis","Arthas","Auriel","Azmodan","Blaze","Brightwing","Cassia","Chen","Cho","Chromie","D.Va","Deckard","Dehaka","Diablo","E.T.C.","Falstad","Fenix",
"Gall","Garrosh","Gazlowe","Genji","Greymane","Gul'dan","Hanzo","Illidan","Jaina","Johanna","Junkrat","Kael'thas","Kel'Thuzad","Kerrigan","Kharazim","Leoric",
"Li Li","Li-Ming","Lt. Morales","Lucio","Lunara","Maiev","Mal'Ganis","Malfurion","Malthael","Medivh","Mephisto","Muradin","Murky","Nazeebo","Nova",
"Orphea","Probius","Ragnaros","Raynor","Rehgar","Rexxar","Samuro","Sgt. Hammer","Sonya","Stitches","Stukov","Sylvanas","Tassadar","The Butcher",
"The Lost Vikings","Thrall","Tracer","Tychus","Tyrael","Tyrande","Uther","Valeera","Valla","Varian","Whitemane","Xul","Yrel","Zagara","Zarya","Zeratul","Zul'jin"]

heroObject = {}
for hero in heroes:
	heroObject[ hero ] = hero
	
heroObject[ "Morales"] = "Lt. Morales"
heroObject["Queen of blades"] = "Kerrigan"
heroObject["Deevah"] = "D.Va"
heroObject["Kaletas"] = "Kael'thas"		
heroObject["A noob arak"] = "Anubarak"		
heroObject["Let da killing begin"] = "Zul'jin"
heroObject["The Lich Lord of The Plaguelands"] = "Kel'thuzad"
heroObject["Commander of The Dread Necropolis"] = "Kel'thuzad"
heroObject["Founder of the Cult of The Damned"] = "Kel'thuzad"
heroObject["Former Member of the Council of Six"] = "Kel'thuzad"
heroObject["Creator of the Abominations"] = "Kel'thuzad"
heroObject["Betrayer of Humanity"] = "Kel'thuzad"
heroObject["Summoner of Archimonde"] = "Kel'thuzad"
heroObject["Majordomo to The Lich King himself"] = "Kel'thuzad"
heroObject["You wot mate"] = "Tracer"
heroObject["Death comes"] = "Malthael"
heroObject["Acceptable outcome"] = "Abathur"
heroObject["By fire be purged"] = "Ragnaros"
heroObject["I am malganis I am a turtle"] = "Mal'ganis"
heroObject["Stitches want to play"] = "Stitches"
heroObject["Essence"] = "Dehaka"
heroObject["Stay a while and listen"] = "Deckard"
heroObject["Do not fail me again"] = "Alarak"
heroObject["The great outdoors"] = "Lunara"
heroObject["Ome wa mo shinderu"] = "Genji"
heroObject["Nandato"] = "Hanzo"
heroObject["My destiny is my own"] = "Illidan"
heroObject["All shall suffer"] = "Leoric"
heroObject["Here we go again"] = "Chromie"
heroObject["Stratholme deserved it"] = "Arthas"
heroObject["Theramore deserved it"] = "Garrosh"
heroObject["Teldrassil deserved it"] = "Sylvanas"
heroObject["Burn it"] = "Sylvanas"
heroObject["Salami all shall adore it"] = "Kael'thas"
heroObject["I play to win"] = "D.Va"
heroObject["Cigar boie"] = "Tychus"
heroObject["There is always hope"] = "Auriel"
heroObject["I bring life and "] = "Alexstrasza"
heroObject["Proxy stargate"] = "Probius"
heroObject["Ha ha ha"] = "Lunara"
heroChoice = Choice( "hero", heroObject)
		
class SelectHeroRule(CompoundRule):
	spec = "<hero>"
	extras = [heroChoice]
	callback = False
	
	def set_callback( self, callback ):
		self.callback = callback

	def _process_recognition(self, node, extras):
		hero = extras["hero"]
		if( self.callback ):
			self.callback( hero )
		