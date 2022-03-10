import pandas as pd

def create_country_dict(isourl, sheetno):
    isos = pd.read_excel(isourl, sheetno)
    isos.columns = isos.iloc[0]
    isos = isos[1:]
    isos = [iso for iso in isos.ISOAlpha]
    isos[0:10]

    countries = pd.read_excel(isourl, sheetno)
    countries.columns = countries.iloc[0]
    countries = countries[1:]
    countries = [country for country in countries['Country or Territory Name']]
    countries[0:10]

    countrydict_ = {}
    for country, iso in zip(countries, isos):
        countrydict_[country] = iso
    countrydict_

    not_found = ['ALA', 'GGY', 'JEY', 'KOS', 'NFK', 'PCN', 'SJM']
    countrydict = {}

    for country, iso in countrydict_.items():
        if iso not in not_found:
            countrydict[country] = iso

    countries = pd.DataFrame(countrydict.keys())
    return countries
