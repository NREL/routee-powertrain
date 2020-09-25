'''
Placeholder variables for default fleet compositions
for Columbus, D.C., and the United States. Values
obtained from the 2014-2017 IHS Markit (formerly Polk)
data of all vehicle registrations in the United States.
These registrations were grouped on the fuel types 
that are supported by RouteE. Each year had an 'Other'
category that consists of FFVs, hydrogen, ethanol, 
methanol, CNG, and propane vehicles. The vast majority
of this category falls under FFV. FFVs are most closely 
comparable to conventional gasoline vehicles. Therefore,
the other category is added to the 'cv_gas' category for
simplicity. Once RouteE has models to represent the other
fuel types, these variables will be adjusted.
'''

NATIONAL_FLEET = {
    '2014': {
        'cv_gas': .9551, #.8859 + .0692
        'cv_diesel': .0319,
        'phev_cd': .0003,
        'phev_cs': .0003,
        'hev': .0118,
        'bev': .0005}, #'Other': '6.92%'},
    '2015': {
        'cv_gas': .9532, #.8789 + .0743
        'cv_diesel': .0325,
        'phev_cd': .00035,
        'phev_cs': .00035,
        'hev': .0127,
        'bev': .0007}, #'Other': '7.43%'},
    '2016': {
        'cv_gas': .9539,#.8792 + .0747
        'cv_diesel': .0300,
        'phev_cd': .0005,
        'phev_cs': .0005,
        'hev': .0141,
        'bev': .0010},#'Other': '7.47%'},
    '2017': {
        'cv_gas': .9518, #.8739 + .0779
        'cv_diesel': .0308,
        'phev_cd': .00065,
        'phev_cs': .00065,
        'hev': .0148,
        'bev': .0013},#'Other': '7.79%'}
}

COLUMBUS_FLEET = {
    '2014': {
        'cv_gas': .9765, #.9177 + .0588
        'cv_diesel': .0114,
        'phev_cd': .0002,
        'phev_cs': .0002,
        'hev': .0114,
        'bev': .0003},#'Other': '5.88%'},    
    '2015': {
        'cv_gas': .9745, #.9102 + .0643
        'cv_diesel': .0118,
        'phev_cd': .0003,
        'phev_cs': .0003,
        'hev': .0127,
        'bev': .0004},#'Other': '6.43%'},   
    '2016': {
        'cv_gas': .9757, #.9100 + .0657
        'cv_diesel': .0101,
        'phev_cd': .0003,
        'phev_cs': .0003,
        'hev': .0131,
        'bev': .0004},#'Other': '6.57%'},    
    '2017': {
        'cv_gas': .9755, #.9076 + .0679
        'cv_diesel': .0089,
        'phev_cd': .0003,
        'phev_cs': .0003,
        'hev': .0142,
        'bev': .0007}, #'Other': '6.79%'} 
}

DC_FLEET = {
    '2014': {
        'cv_gas': .9585, #.9138 + .0447
        'cv_diesel': .0125,
        'phev_cd': .0005,
        'phev_cs': .0005,
        'hev': .0272,
        'bev': .0008},#'Other': '4.47%'},
    '2015': {
        'cv_gas': .9549, #.9039 + .0510
        'cv_diesel': .0131,
        'phev_cd': .0006,
        'phev_cs': .0006,
        'hev': .0296,
        'bev': .0012},#'Other': '5.10%'},
    '2016': {
        'cv_gas': .9477, #.8954 + .0523
        'cv_diesel': .0127,
        'phev_cd': .00085,
        'phev_cs': .00085,
        'hev': .0363,
        'bev': .0016},#'Other': '5.23%'},
    '2017': {
        'cv_gas': .9474, #.8923 + .0551
        'cv_diesel': .0108,
        'phev_cd': .00115,
        'phev_cs': .00115,
        'hev': .0376,
        'bev': .0020}#'Other': '5.51%'}
}