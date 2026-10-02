"""Training-only, temperature-scaled KL on legal outputs, guarded by gold targets."""
import torch
import torch.nn.functional as F

def distillation_loss(student,teacher,targets,weights,temperature=2.):
    # The deployed gate chooses only KNOWN/UNKNOWN. Answers choose EOS or text.
    # Keep ordinary supervised CE elsewhere; do not imitate a teacher's wrong
    # argmax, including wrong refusal decisions, even under teacher forcing.
    total=student.sum()*0.;agree=weights.new_zeros(());mass=weights.sum()
    for gate in [True,False]:
        positions=(weights>0)&((targets==5)|(targets==6) if gate else (targets!=5)&(targets!=6))
        s=student[positions];t=teacher[positions].detach();y=targets[positions];w=weights[positions]
        if not len(y):continue
        if gate:s=s[:,5:7];t=t[:,5:7];y=y-5
        else:
            legal=torch.cat([torch.tensor([4],device=s.device),torch.arange(7,1024,device=s.device)])
            s=s[:,legal];t=t[:,legal];y=torch.where(y==4,0,y-6)
        correct=t.argmax(-1)==y
        kl=F.kl_div(F.log_softmax(s/temperature,dim=-1),F.softmax(t/temperature,dim=-1),reduction='none').sum(-1)
        total=total+(kl*w*correct).sum()*temperature**2;agree=agree+(w*correct).sum()
    return total/mass,agree/mass
