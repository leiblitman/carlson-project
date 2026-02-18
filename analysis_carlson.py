import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import warnings, re
warnings.filterwarnings('ignore')

DATA = 'c:/Users/leibl/Documents/PositronProjects/Carlson Project/data/raw/result_persuasion_PrimePanels_2.17.26.csv'
df = pd.read_csv(DATA)
print('Raw rows loaded:', len(df))
if 'Interview Status' in df.columns:
    statuses = df['Interview Status'].value_counts().to_dict()
    print('Interview statuses:', statuses)
    n_completed = statuses.get('Completed', 0)
    print('NOTE: n_completed =', n_completed, ' -- using ALL rows for analysis given pilot size')
N = len(df)
SEP = '=' * 72

def section(t):
    print()
    print(SEP)
    print('  ' + str(t))
    print(SEP)

def pct_dist(s):
    vc = s.value_counts(dropna=True)
    tot = vc.sum()
    out = {}
    for k2,v2 in vc.items():
        out[k2] = str(v2) + ' (' + str(round(100*v2/tot,1)) + '%)'
    return out

def mean_ci(s):
    s2 = pd.to_numeric(s, errors='coerce').dropna()
    if len(s2)==0: return 'N/A'
    return str(round(s2.mean(),2)) + ' (SD=' + str(round(s2.std(),2)) + ', n=' + str(len(s2)) + ')'

def get_vc(qnum):
    for c in df.columns:
        if '(' + str(qnum) + ')' in c and c.endswith('- Value'): return c

def short_label(c):
    m = re.search(r'[(]([0-9]+[.][0-9]+)[)]', c)
    qn = m.group(1) if m else '?'
    parts = c.split(' - ')
    phrase = parts[-2] if len(parts)>=2 else c
    return 'Q' + qn + ': ' + phrase[:38]

age_col='(26) Option Code'
gen_col='(27) Option Code'
race_col='(29) Option Code'
ideo_col='(30) Option Code'
party_col='(31) Option Code'
income_col='(32) Option Code'
edu_col='(33) Option Code'
state_col='(35) In what state do you currently reside?'

# SECTION 1
section('1. SAMPLE CHARACTERISTICS')
print('Total respondents (all statuses):', N)
if 'Interview Status' in df.columns:
    print('  Status breakdown:')
    for k2,v2 in pct_dist(df['Interview Status']).items(): print('    ' + str(k2) + ': ' + str(v2))
for label,colname in [('Age',age_col),('Gender',gen_col),('Race/Ethnicity',race_col),
    ('Political Ideology',ideo_col),('Party Affiliation',party_col),
    ('Household Income',income_col),('Education',edu_col)]:
    if colname in df.columns:
        print()
        print(label + ':')
        for k2,v2 in pct_dist(df[colname]).items(): print('  ' + str(k2) + ': ' + str(v2))
if state_col in df.columns:
    print()
    print('Top 15 States:')
    for st,cnt in df[state_col].value_counts().head(15).items():
        print('  ' + str(st) + ': ' + str(cnt) + ' (' + str(round(100*cnt/N,1)) + '%)')

# SECTION 2
section('2. TUCKER CARLSON FAMILIARITY & ENGAGEMENT')
watch_col='(3) Option Code'
fam_col='(4) Option Code'
if watch_col in df.columns:
    print('Q3 - Watch/listen frequency:')
    for k2,v2 in pct_dist(df[watch_col]).items(): print('  ' + str(k2) + ': ' + str(v2))
if fam_col in df.columns:
    print()
    print('Q4 - Familiarity with Carlson Israel views:')
    for k2,v2 in pct_dist(df[fam_col]).items(): print('  ' + str(k2) + ': ' + str(v2))
print()
print('Q1 - Other commentator familiarity (% of sample):')
comm = {
    'Tucker Carlson':'(1) Tucker Carlson','Rachel Maddow':'(1) Rachel Maddow',
    'Cenk Uygur':'(1) Cenk Uygur','Ben Shapiro':'(1) Ben Shapiro',
    'Candace Owens':'(1) Candace Owens','Hasan Piker':'(1) Hasan Piker',
    'Megyn Kelly':'(1) Megyn Kelly','Marc Lamont Hill':'(1) Marc Lamont Hill',
    'Scott Jennings':'(1) Scott Jennings','Marcus Thornwell':'(1) Marcus Thornwell'}
for name,col in comm.items():
    if col in df.columns:
        k3=df[col].notna().sum()
        print('  ' + name.ljust(22) + ': ' + str(k3) + ' (' + str(round(100*k3/N,1)) + '%)')
print()
print('Q2 - Should be listened to (0-100):')
q2m = {
    'Tucker Carlson':'(2.1) How would you characterize each of the following commentators? Use the slider to select a number from 0 (Should be ignored ) to 100 (Should be listened to). - Tucker Carlson - Value',
    'Cenk Uygur':'(2.2) How would you characterize each of the following commentators? Use the slider to select a number from 0 (Should be ignored ) to 100 (Should be listened to). - Cenk Uygur - Value',
    'Ben Shapiro':'(2.3) How would you characterize each of the following commentators? Use the slider to select a number from 0 (Should be ignored ) to 100 (Should be listened to). - Ben Shapiro - Value',
    'Hasan Piker':'(2.4) How would you characterize each of the following commentators? Use the slider to select a number from 0 (Should be ignored ) to 100 (Should be listened to). - Hasan Piker - Value',
    'Scott Jennings':'(2.5) How would you characterize each of the following commentators? Use the slider to select a number from 0 (Should be ignored ) to 100 (Should be listened to). - Scott Jennings - Value'}
for name,col in q2m.items():
    if col in df.columns: print('  ' + name.ljust(22) + ': ' + mean_ci(df[col]))

# SECTION 3
section('3. CONSPIRACY BELIEFS (Q7)')
q7_items = {
    'Vaccines dangerous for children':'(7.1) Please indicate your level of agreement with the following statements: - Vaccines are dangerous for children - Code',
    'Moon landing was staged':'(7.2) Please indicate your level of agreement with the following statements: - The moon landing was probably staged - Code',
    'Epstein was a Mossad agent':'(7.3) Please indicate your level of agreement with the following statements: - Jeffrey Epstein was a Mossad agent - Code',
    'Govt hiding alien contact':'(7.4) Please indicate your level of agreement with the following statements: - The government is hiding evidence of alien contact - Code',
    '2020 election stolen':'(7.5) Please indicate your level of agreement with the following statements: - The 2020 election was stolen from Donald Trump - Code'}
for label,col in q7_items.items():
    if col in df.columns:
        print()
        print('  ' + label + ':')
        for k2,v2 in pct_dist(df[col]).items(): print('    ' + str(k2) + ': ' + str(v2))

# SECTIONS 4-8: pre/post helper
def pre_post_table(sec_title, item_map):
    section(sec_title)
    print('  ' + 'Item'.ljust(42) + 'Pre'.rjust(8) + 'Post'.rjust(8) + 'Diff'.rjust(8) + 'p'.rjust(10))
    print('  ' + '-'*76)
    pre_cols=[]; post_cols=[]
    for label,(pq,oq) in item_map.items():
        pc=get_vc(pq); oc=get_vc(oq)
        if pc and oc:
            pre=pd.to_numeric(df[pc],errors='coerce')
            post=pd.to_numeric(df[oc],errors='coerce')
            mask=pre.notna()&post.notna()
            if mask.sum()>1:
                _t,p=stats.ttest_rel(pre[mask],post[mask])
                d=post[mask].mean()-pre[mask].mean()
                sig='**' if p<.01 else ('*' if p<.05 else ('+' if p<.10 else ' '))
                pm=round(pre[mask].mean(),2); om=round(post[mask].mean(),2)
                dr=round(d,2); pr=round(p,4)
                dsign='+' if d>=0 else ''
                print('  ' + label.ljust(42) + str(pm).rjust(8) + str(om).rjust(8) + (dsign+str(abs(dr))).rjust(8) + str(pr).rjust(10) + sig)
                pre_cols.append(pc); post_cols.append(oc)
    print('  ** p<.01  * p<.05  + p<.10')
    return pre_cols, post_cols

cred_pre,cred_post=pre_post_table('4. CREDIBILITY RATINGS (Q8=PRE, Q18=POST)',
    {'Trust foreign policy analysis':('8.1','18.1'),'Gets facts right':('8.2','18.2'),
     'Predictions accurate':('8.3','18.3'),'Covers ignored topics':('8.4','18.4'),
     'Challenges mainstream narratives':('8.5','18.5'),'Often gets things wrong':('8.6','18.6')})

fp_pre,fp_post=pre_post_table('5. FOREIGN POLICY VIEWS (Q9=PRE, Q19=POST)',
    {'Russia provoked by NATO/West':('9.1','19.1'),
     'Israel/AIPAC too much influence':('9.2','19.2'),
     'Cut all foreign funding':('9.3','19.3')})

therm_pre,therm_post=pre_post_table('6. FEELINGS THERMOMETERS (Q10=PRE, Q20=POST)',
    {'Israeli people':('10.1','20.1'),'Jewish people':('10.2','20.2'),
     'State of Israel':('10.3','20.3')})

toi_pre,toi_post=pre_post_table('7. TUCKER ON ISRAEL (Q11=PRE, Q21=POST)',
    {'Commentary accurate and fair':('11.1','21.1'),
     'Asks right questions (US-Israel)':('11.2','21.2'),
     'Too critical of Israel':('11.3','21.3'),
     'Understands Middle East better':('11.4','21.4')})

ip_pre,ip_post=pre_post_table('8. ISRAEL POLICY VIEWS (Q12=PRE, Q22=POST)',
    {'US-Israel relationship benefits US':('12.1','22.1'),
     'Israel right to exist as Jewish state':('12.2','22.2'),
     'Israel acts morally in self-defense':('12.3','22.3'),
     'US should reduce military aid':('12.4','22.4')})

# SECTION 9
section('9. K-MEANS CLUSTERING ON PRE-TREATMENT ATTITUDE ITEMS')
q7_code_cols=[c for c in df.columns if c.startswith('(7.') and c.endswith('- Code')]
cluster_cols=q7_code_cols+fp_pre+therm_pre+toi_pre+ip_pre
cdf2=df[cluster_cols].apply(pd.to_numeric,errors='coerce')
# impute missing with column median for clustering
cdf2_imp=cdf2.copy()
for col in cluster_cols:
    cdf2_imp[col]=cdf2_imp[col].fillna(cdf2_imp[col].median())
cdata=cdf2_imp.copy()
dfull=df.copy()
n_complete=cdf2.notna().all(axis=1).sum()
print('Rows with fully complete pre-treatment data:', n_complete, '/', N)
print('Using median-imputed data for clustering, n=', len(cdata))
print('Variables (' + str(len(cluster_cols)) + '):')
for c in cluster_cols: print('  ' + short_label(c))
scaler=StandardScaler()
X=scaler.fit_transform(cdata)
sil_scores={}
for k in [2,3,4]:
    if k>=len(cdata): continue
    km=KMeans(n_clusters=k,n_init=20,random_state=42)
    labs=km.fit_predict(X)
    if len(set(labs))>1:
        sil=silhouette_score(X,labs)
    else:
        sil=0.0
    sil_scores[k]={'labels':labs,'sil':sil}
    sizes=dict(zip(*np.unique(labs,return_counts=True)))
    print('  k=' + str(k) + ': Silhouette=' + str(round(sil,4)) + '  |  Cluster sizes: ' + str(sizes))
omeans=cdata[cluster_cols].mean()
clbls=[short_label(c) for c in cluster_cols]
for k in list(sil_scores.keys()):
    divl='-'*72
    sil_val=sil_scores[k]['sil']
    print()
    print(divl)
    print('  CLUSTER PROFILES  k=' + str(k) + '  (Silhouette=' + str(round(sil_val,4)) + ')')
    print(divl)
    labs=sil_scores[k]['labels']
    cd2=cdata.copy(); cd2['_cl']=labs
    dd2=dfull.copy(); dd2['_cl']=labs
    for cl in range(k):
        mc=cd2['_cl']==cl; nc=mc.sum()
        cm=cd2[mc][cluster_cols].mean()
        print()
        print('  --- Cluster ' + str(cl+1) + '  (n=' + str(nc) + ', ' + str(round(100*nc/len(cdata),1)) + '%) ---')
        print('  ' + 'Variable'.ljust(50) + 'Mean'.rjust(8) + 'vs Overall'.rjust(12))
        for cn,lb in zip(cluster_cols,clbls):
            m=cm[cn]; ov=omeans[cn]; d=m-ov
            flag=' <<' if d<-8 else (' >>' if d>8 else '')
            dsign='+' if d>=0 else ''
            print('    ' + lb.ljust(48) + str(round(m,1)).rjust(8) + (dsign+str(round(abs(d),1))).rjust(12) + flag)
        ds=dd2[dd2['_cl']==cl]
        if ideo_col in ds.columns:
            print('  Ideology:', ds[ideo_col].value_counts(normalize=True).mul(100).round(1).to_dict())
        if party_col in ds.columns:
            print('  Party:   ', ds[party_col].value_counts(normalize=True).mul(100).round(1).to_dict())

# SECTION 10
section('10. QUALITATIVE THEMES')
q5col='(5) Why do you watch or listen to Tucker Carlson?'
if q5col in df.columns:
    q5=df[q5col].dropna()
    print('Q5 - Why watch Tucker Carlson?  (n=' + str(len(q5)) + ' responded)')
    print()
    print('  Sample responses (first 25):')
    for i,r in enumerate(q5.head(25),1):
        snippet=str(r)[:120].replace(chr(10),' ')
        print('  ' + str(i).rjust(2) + '. ' + snippet)
    kws={
        'honest / truth':['honest','honesty','truth','truthful','real','genuine'],
        'anti-mainstream media':['mainstream','msm','establishment','fake news'],
        'challenges narratives':['question','challenge','critical','pushback','debunk'],
        'information / learn':['news','inform','information','learn','update'],
        'entertainment':['entertain','funny','humor','amusing','interesting'],
        'foreign policy':['foreign policy','nato','ukraine','russia'],
        'Israel / Middle East':['israel','jewish','middle east','aipac','gaza'],
        'shares my views':['agree','share','align','conservative','like-minded'],
        'free speech / courage':['censor','brave','courage','says what','free speech'],
        'dont watch':['dont watch','never watch','not familiar','do not watch']}
    print()
    print('  Theme frequencies in Q5:')
    for theme,terms in kws.items():
        cnt=sum(1 for r in q5 if any(t in str(r).lower() for t in terms))
        print('    ' + theme.ljust(42) + ': ' + str(cnt).rjust(4) + ' (' + str(round(100*cnt/max(len(q5),1),1)) + '%)')

for qnum,qlabel in [('15','Q15'),('16','Q16')]:
    qcol='(' + qnum + ') What do you think Tucker Carlson gets right about Israel?'
    if qcol in df.columns:
        qr=df[qcol].dropna()
        if len(qr)==0: continue
        print()
        print(qlabel + ' - Tucker gets right about Israel?  (n=' + str(len(qr)) + ' responded)')
        print()
        print('  Sample responses (first 25):')
        for i,r in enumerate(qr.head(25),1):
            snippet=str(r)[:130].replace(chr(10),' ')
            print('  ' + str(i).rjust(2) + '. ' + snippet)
        kws2={
            'AIPAC / lobby':['aipac','lobby','lobbying'],
            'undue influence / control':['influence','control','power','sway'],
            'foreign aid / money':['foreign aid','fund','money','billion','dollar'],
            'war / conflict':['war','conflict','military','genocide','attack','bomb'],
            'America first':['america first','national interest','our country'],
            'asks questions':['question','ask','raise','bring up'],
            'media censorship':['media','censorship','censor','suppress','silence'],
            'nothing / not sure':['nothing','not much','unsure','cannot']}
        print()
        print('  Theme frequencies in ' + qlabel + ':')
        for theme,terms in kws2.items():
            cnt=sum(1 for r in qr if any(t in str(r).lower() for t in terms))
            if cnt>0: print('    ' + theme.ljust(42) + ': ' + str(cnt).rjust(4) + ' (' + str(round(100*cnt/len(qr),1)) + '%)')

q13col='(13) Option Code'
if q13col in df.columns:
    print()
    print('Q13 - Has Tucker influenced your views on Israel?')
    for k2,v2 in pct_dist(df[q13col]).items(): print('  ' + str(k2) + ': ' + str(v2))

for qtxt in [
    '(23) Did anything in the conversation you just had change how you think about Tucker Carlson?',
    '(24) Did anything in the conversation you just had change how you think about Israel?']:
    if qtxt in df.columns:
        qr2=df[qtxt].dropna()
        if len(qr2)==0: continue
        print()
        print(qtxt[:80] + '  (n=' + str(len(qr2)) + ')')
        for i,r in enumerate(qr2.head(10),1):
            print('  ' + str(i).rjust(2) + '. ' + str(r)[:120].replace(chr(10),' '))

print()
print(SEP)
print('  ANALYSIS COMPLETE')
print(SEP)
