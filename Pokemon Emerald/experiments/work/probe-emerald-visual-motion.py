"""Learn action-conditioned translations from raw training pixels and score them on later unseen pairs."""
from collections import Counter,defaultdict
import json
from pathlib import Path
import sys
import numpy as np

project=Path(__file__).resolve().parents[1]/"outputs/EmeraldRL"
sys.path.insert(0,str(project))
from train_earned_dreamer import Trajectory,write_json


def translation(before,after):
    # Estimate generic 2-D image translation. No coordinates, map sets or input-button semantics.
    a=before.astype(np.float32).mean(axis=2)
    b=after.astype(np.float32).mean(axis=2)
    change=np.abs(b-a).mean()
    if change<2.5:
        return None
    product=np.fft.rfft2(b-b.mean())*np.conj(np.fft.rfft2(a-a.mean()))
    corr=np.fft.irfft2(product/(np.abs(product)+1e-5),s=a.shape,axes=(0,1))
    y,x=np.unravel_index(np.argmax(corr),a.shape)
    h,w=a.shape
    y=int((y+h//2)%h-h//2)
    x=int((x+w//2)%w-w//2)
    if (x==0 and y==0) or abs(x)>w//4 or abs(y)>h//4:
        return None
    top,bottom=max(0,y)+1,min(h,h+y)-1
    left,right=max(0,x)+1,min(w,w+x)-1
    observed=b[top:bottom,left:right]
    residual=np.abs(observed-a[top-y:bottom-y,left-x:right-x]).mean()
    unshifted=np.abs(observed-a[top:bottom,left:right]).mean()
    if residual>.55*unshifted or unshifted<2.5:
        return None
    return x,y


def main(dataset,out):
    game=Trajectory(dataset)
    split=len(game.tape)*3//4
    train,held=defaultdict(Counter),[]
    hist=Counter()
    previous=None
    for i,(frame,button,reward,first,last,terminal) in enumerate(game.episode()):
        if previous is not None:
            observed=translation(previous,frame)
            hist["pairs"]+=1
            if observed is not None:
                hist["confident_pairs"]+=1
                if i<=split:
                    train[priorbutton][observed]+=1
                else:
                    held.append((priorbutton,observed))
        previous,priorbutton=frame,button
    choices={}
    for button, counts in train.items():
        mode,n=counts.most_common(1)[0]
        if n>=30 and n/sum(counts.values())>=.60:
            choices[button]=mode
    eligible=[(button,label) for button,label in held if button in choices]
    if not eligible:
        raise ValueError("No confident held-out movements from a trained action")
    majority=max(Counter(x for _,x in eligible).values())/len(eligible)
    score=sum(choices[a]==x for a,x in eligible)/len(eligible)
    per_button={str(a):dict(train_samples=sum(train[a].values()),most_common=list(choices[a]),
        heldout_samples=sum(b==a for b,x in eligible),heldout_correct=sum(b==a and choices[a]==x for b,x in eligible))
        for a in choices}
    out.parent.mkdir(parents=True,exist_ok=True)
    write_json(out,dict(training_tape_sha256=game.source["source_sha256"],train_through_button=split,
        heldout_trajectory_positions=len(game.tape)-split,total_pairs=hist["pairs"],confident_pairs=hist["confident_pairs"],
        heldout_confident_motion_pairs=len(held),eligible_heldout_pairs=len(eligible),conditional_accuracy=score,
        majority_accuracy=majority,learned_actions=per_button,position_or_map_input=False,
        informative=(len(eligible)>=100 and len(choices)>=2 and score-majority>=.2)))
    print(json.dumps(dict(eligible=len(eligible),accuracy=score,majority=majority,actions=per_button)),flush=True)


if __name__=="__main__":
    main(Path(sys.argv[1]),Path(sys.argv[2]))
