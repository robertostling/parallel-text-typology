from langinfo.glottolog import Glottolog

family_iso = {}
def family_from_iso(iso):
	family = family_iso.get(iso)
	if family is None:
		family = Glottolog[iso].family_id
		if family is None:
			family = Glottolog[iso].id
		family_iso[iso] = family
	return family

