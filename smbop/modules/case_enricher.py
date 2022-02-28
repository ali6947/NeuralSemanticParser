import torch

class Enricher():
	def __init__(self,emb_dim):
		self.schema_schema_attn=torch.nn.MultiheadAttention(emb_dim,1)
		self.schema_utt_attn=torch.nn.MultiheadAttention(emb_dim,1)

	def _get_case_enriched_schema(self,utt,utt_mask,schema,schema_mask):
        # utt : b x C x T x D
        # utt_mask: b x C x T
        # schema : b x C x E x D
        # schema_mask : b x C x E
        #True means not masked and false is masked so using * combines two masks.
        b,C,T,D=utt.shape
        b1,c1,E,d1=schema.shape
        assert b==b1 and C==c1 and D==d1
       
        schema_for_attn=schema.transpose(1,0).reshape((C,-1,D)) # C x b*E x D
        schema_mask_for_attn=schema_mask.transpose(2,1).reshape((-1,C)) # b*E x C, torch needs mask to be batch_size x seq_len
        schema_schema_attn_op=self.schema_schema_attn(schema_for_attn,schema_for_attn,schema_for_attn,torch.logical_not(schema_mask_for_attn)) # b*E x C x C
        schema_schema_align_score=schema_schema_attn_op[1].reshape((-1,E,C,C)).transpose(1,2) # b x C x E x C . THis is P(s_c|s_c',cases) first C(dim=1) corresponds to s_c'(schema item s in case c'), dim=3 to s_c because
        # schema_attn_op is of shape b*E,C,C where first C denotes the iteration over queries and queries are s_c
        schema_schema_align_mask=(schema_mask.unsqueeze(2)*schema_mask.unsqueeze(1)).transpose(2,3) # b x C x E x C. Note that element (i,j,k,l) = schema_mask[i,j,l] && schema_mask[i,k,l]

        utt_for_attn=utt.reshape((-1,T,D)).transpose(1,0) # T x bC x D
        rep_schema=schema.reshape((-1,C*E,D)).transpose(1,0).repeat(1,1,C).reshape((E*C,b*C,D)) # after each op shape varies as(b,CE,D)->(CE,b,D)->(CE,b,CD)->(EC,bC,D)
        utt_mask_for_attn=utt_mask.reshape((-1,T)) # bC x T
        schema_utt_attn_op=self.schema_utt_attn(rep_schema,utt_for_attn,utt_for_attn,torch.logical_not(utt_mask_for_attn)) # the first element as shape bC x EC x T
        schema_utt_align_score=schema_utt_attn_op[1].reshape((b,C,E,C,T)) # note that the first C corresponds to the 'values' input of attn layer since it came from the dim=0 
        # and so controls x_j in P(x_j | x , z(s_c)). The second C controls c in z(s_c)
        schema_utt_align_mask=(schema_mask.unsqueeze(3).unsqueeze(3)*utt_mask.unsqueeze(1).unsqueeze(1)).transpose(1,3) # b x C x E x C x T.
        # Entry at loc(i,j,k,l,m)=schema_mask(i,l,k) && utt_mask(i,j,m)
        alignment_probs=(schema_utt_align_score*schema_utt_align_mask)*((schema_schema_align_score*schema_schema_align_mask).unsqueeze(4)) # b x C x E x C x T.
        # This gives P(x_j|s_c,x)*P(s_c|s_c')  The first c of schema_utt_align_score defines the x we are studying and the first c of schema_schema_align_score defines the s_c' under consdirations
        # for which we have the probs so c' and x correspond to the same thing
        weighted_schema_utts=(alignment_probs.unsqueeze(5)*utt.unsqueeze(2).unsqueeze(2)) # b x C x E x C x T x D

        enriched_schema=weighted_schema_utts.sum(3).sum(3) # b x C x E x D

        assert enriched_schema.shape==schema.shape
        
        # print(f'schema_for_attn: {schema_for_attn.shape}')
        # print(f'schema_mask_for_attn: {schema_mask_for_attn.shape}')
        # print(f'schema_schema_align_score: {schema_schema_align_score.shape}')
        # print(f'schema_schema_align_mask: {schema_schema_align_mask.shape}')
        # print(f'utt_for_attn: {utt_for_attn.shape}')
        # print(f'rep_schema: {rep_schema.shape}')
        # print(f'utt_mask_for_attn: {utt_mask_for_attn.shape}')
        # print(f'schema_utt_align_score: {schema_utt_align_score.shape}')
        # print(f'schema_utt_align_mask: {schema_utt_align_mask.shape}')
        # print(f'alignment_probs: {alignment_probs.shape}')
        # print(f'weighted_schema_utts: {weighted_schema_utts.shape}')
        # print(f'enriched_utt: {enriched_utt.shape}')

        return enriched_schema



if __name__=='__main__':
	
	b=2
	C=3
	E=4
	D=5
	T=6
	a=Enricher(D)
	utt=torch.rand((b,C,T,D))
	utt_mask=torch.rand((b,C,T))>0.5
	schema=torch.rand((b,C,E,D))
	schema_mask=torch.rand((b,C,E))>0.5
	a._get_case_enriched_schema(utt,utt_mask,schema,schema_mask)