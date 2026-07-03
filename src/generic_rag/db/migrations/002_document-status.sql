alter table documents
    add column status varchar(15) not null default 'created';
